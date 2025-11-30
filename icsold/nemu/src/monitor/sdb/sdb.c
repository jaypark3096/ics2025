/***************************************************************************************
* Copyright (c) 2014-2024 Zihao Yu, Nanjing University
*
* NEMU is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
*
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
*
* See the Mulan PSL v2 for more details.
***************************************************************************************/

#include <isa.h>
#include <cpu/cpu.h>
#include <readline/readline.h>
#include <readline/history.h>
#include "sdb.h"
#include <memory/paddr.h>
#include <stdint.h>
static int is_batch_mode = false;

void init_regex();
void init_wp_pool();



/* We use the `readline' library to provide more flexibility to read from stdin. */
static char* rl_gets() {
  static char *line_read = NULL;

  if (line_read) {
    free(line_read);
    line_read = NULL;
  }

  line_read = readline("(nemu) ");

  if (line_read && *line_read) {
    add_history(line_read);
  }

  return line_read;
}

static int cmd_c(char *args) {
  cpu_exec(-1);
  return 0;
}


static int cmd_q(char *args) {
    clear_history();
    exit(0);
  return 0;
}

static int cmd_help(char *args);

static int cmd_si(char* args) {
    char* arg = strtok(NULL, " ");
    uint64_t n = 1;     // 默认单步调试
    if (arg != NULL) {  // si [N] 中的[N]不为空
#if __WORDSIZE == 64
        n = atol(arg);
#else
        n = atoll(arg);
#endif
    }
    cpu_exec(n);
    return 0;
}


// 设置监视点
static int cmd_w(char* args) {
    if (args == NULL) {
        printf("Usage: w <expression>\n");
        return 0;
    }

    WP* wp = new_wp(args);
    if (wp != NULL) {
        printf("Watchpoint %d set: %s (0x%x)\n", wp->NO, args, wp->value);
    }
    return 0;
}

// 删除监视点
static int cmd_dwp(char* args) {
    if (args == NULL) {
        printf("Usage: dwp <watchpoint-number>\n");
        return 0;
    }

    int no = atoi(args);
    if (free_wp(no)) {
        printf("Watchpoint %d deleted\n", no);
    }
    return 0;
}


static int cmd_info(char* args) {
    char* arg = strtok(NULL, " ");
    if (arg == NULL) {
        printf("Usage: info <subcommand>\n");
        printf("Subcommands:\n  r -- Show registers\n  w -- Show watchpoints\n");
        return 0;
    }
    else if (arg[0] == 'r') {
        isa_reg_display();
    }
    else if (arg[0] == 'w') {
        WP* wp = get_wp_head();
        if (wp == NULL) {
            printf("No watchpoints set\n");
            return 0;
        }

        printf("%-6s %-32s %s\n", "Num", "Expression", "Value");
        printf("-----------------------------------------------------\n");
        while (wp != NULL) {
            
            printf("%-6d %-32s 0x%x\n", wp->NO, wp->expr, wp->value);
            wp = wp->next;
        }
    }
    else {
        printf("Unknown info subcommand: %s\n", arg);
    }
    return 0;
}


static int cmd_x(char* args) {
    char* arg = strtok(NULL, " ");
    if (arg == NULL) {
        printf("Argument required (starting display address).\n");
        return 0;
    }
    int64_t n = 1;
    char* expr = strtok(NULL, " ");
    if (expr != NULL) {
        n = atol(arg);  // n 为负数，向后打印
    }
    else {
        expr = arg;
    }
    if (expr[0] == '0' && (expr[1] == 'x' || expr[1] == 'X')) {
        expr += 2;
    }
    else {
        printf("Invalid number \"%s\".\n", expr);
        return 0;
    }
    size_t len = strlen(expr);
    uint64_t address = 0;
    for (int i = 0; i < len; ++i) {
        address <<= 4;
        if (isdigit(expr[i])) {
            address += (expr[i] - '0');
        }
        else if (expr[i] >= 'a' && expr[i] <= 'f') {
            address += (expr[i] - 'a' + 10);
        }
        else if (expr[i] >= 'A' && expr[i] <= 'F') {
            address += (expr[i] - 'A' + 10);
        }
        else {
            printf("Invalid number \"%s\".\n", expr - 2);
            return 0;
        }
    }
    // 读取数据并打印
    int direct = n > 0 ? 4 : -4;    // 地址增加的方向
    n = n > 0 ? n : -n;
    for (; n > 0; --n) {
        word_t ret = paddr_read(address, 4);  // paddr_read已经做了地址合法性的检查
        printf(ANSI_FMT("0x%lx: ", ANSI_FG_BLUE), address);
        printf("0x%08x\n", ret);   // 4字节，16进制就有8个字符，右对其，高位补0
        address += direct;
    }
    return 0;
}

static int cmd_p(char* args) {
    if (args == NULL) {
        printf("Usage: p <expression>\n");
        return 0;
    }

    bool success;
    word_t result = expr(args, &success);
    if (success) {
        // 同时打印十进制和十六进制，符合调试器惯例
        printf("%lu (0x%lx)\n", (unsigned long)result, (unsigned long)result);
    }
    else {
        printf("Invalid expression: %s\n", args);
    }
    return 0;
}


static struct {
  const char *name;
  const char *description;
  int (*handler) (char *);
} cmd_table [] = {
  { "help", "Display information about all supported commands", cmd_help },
  { "c", "Continue the execution of the program", cmd_c },
  { "q", "Exit NEMU", cmd_q },
  
  /* TODO: Add more commands */
  { "si", "Single-step execution", cmd_si},
  {"info", "List of info subcommands:\n\n"
           "info r -- List of all registers and their contents, for selected stack frame.", cmd_info},
  {"x","Scan memory",cmd_x},
  {"w", "Set watchpoint: w <expression>", cmd_w},
  {"dwp", "Delete watchpoint: dwp <number>", cmd_dwp},
  { "p", "Evaluate expression and print result", cmd_p },


};

#define NR_CMD ARRLEN(cmd_table)

static int cmd_help(char *args) {
  /* extract the first argument */
  char *arg = strtok(NULL, " ");
  int i;

  if (arg == NULL) {
    /* no argument given */
    for (i = 0; i < NR_CMD; i ++) {
      printf("%s - %s\n", cmd_table[i].name, cmd_table[i].description);
    }
  }
  else {
    for (i = 0; i < NR_CMD; i ++) {
      if (strcmp(arg, cmd_table[i].name) == 0) {
        printf("%s - %s\n", cmd_table[i].name, cmd_table[i].description);
        return 0;
      }
    }
    printf("Unknown command '%s'\n", arg);
  }
  return 0;
}

void sdb_set_batch_mode() {
  is_batch_mode = true;
}

void sdb_mainloop() {
    sdb_set_batch_mode();
  if (is_batch_mode) {
    cmd_c(NULL);
    return;
  }

  for (char *str; (str = rl_gets()) != NULL; ) {
    char *str_end = str + strlen(str);

    /* extract the first token as the command */
    char *cmd = strtok(str, " ");
    if (cmd == NULL) { continue; }

    /* treat the remaining string as the arguments,
     * which may need further parsing
     */
    char *args = cmd + strlen(cmd) + 1;
    if (args >= str_end) {
      args = NULL;
    }

#ifdef CONFIG_DEVICE
    extern void sdl_clear_event_queue();
    sdl_clear_event_queue();
#endif

    int i;
    for (i = 0; i < NR_CMD; i ++) {
      if (strcmp(cmd, cmd_table[i].name) == 0) {
        if (cmd_table[i].handler(args) < 0) { 
            free(str);
            return; }
        break;
      }
    }

    if (i == NR_CMD) { printf("Unknown command '%s'\n", cmd); }
  }
}

void init_sdb() {
  /* Compile the regular expressions. */
  init_regex();

  /* Initialize the watchpoint pool. */
  init_wp_pool();
}
