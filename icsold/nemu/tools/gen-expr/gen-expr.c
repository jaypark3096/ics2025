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

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <string.h>

// this should be enough
static char buf[65536] = {};
static char code_buf[65536 + 128] = {}; // a little larger than `buf`
static char *code_format =
"#include <stdio.h>\n"
"int main() { "
"  unsigned result = %s; "
"  printf(\"%%u\", result); "
"  return 0; "
"}";

static uint32_t choose(uint32_t n) {
    assert(n > 0);  // 防止传入0导致除0
    return rand() % n;
}

static void gen_space();  // 提前声明 gen_space 函数

static void gen_num() {
    size_t current_len = strlen(buf);
    char num_str[20] = {};  // 32位无符号数最大10位，足够容纳

    // 1. 生成32位无符号数（rand()默认16位，两次组合成32位）
    // 生成16位小整数（0~65535），相乘结果不会溢出32位无符号整数
    uint32_t num = rand() % 65536;  // 65536是2的16次方，确保数值在0~65535之间
    // 2. 转为字符串（适配无符号格式）
    sprintf(num_str, "%uU", num);
    size_t num_len = strlen(num_str);

    // 3. 检查缓冲区空间：数字+前后空格（最多2个）不超过上限
    if (current_len + num_len + 2 < sizeof(buf)) {
        gen_space();       // 数字前加随机空格
        strcat(buf, num_str);  // 拼接数字
        gen_space();       // 数字后加随机空格
    }
}

static void gen_space() {
    size_t current_len = strlen(buf);
    // 预留1字节给'\0'，确保空间足够才加空格
    if (current_len + 1 < sizeof(buf)) {
        if (choose(2) == 0) {  // 50%概率加空格
            strcat(buf, " ");
        }
    }
}

static void gen_rand_op() {
    // 运算符数组：包含双字符运算符（<<、>>），需用字符串存储
    const char* ops[] = { "+", "+", "-", "-", "*", "/", "/", "&", "&", "|", "|", "^", "^", "<<", ">>" };
    size_t op_cnt = sizeof(ops) / sizeof(ops[0]);  // 运算符数量（9个）
    int op_idx = choose(op_cnt);  // 随机选一个运算符
    const char* op = ops[op_idx];
    size_t op_len = strlen(op);   // 处理双字符运算符（如<<占2字节）
    size_t current_len = strlen(buf);

    // 检查缓冲区空间：运算符+前后空格不超过上限
    if (current_len + op_len + 2 < sizeof(buf)) {
        gen_space();       // 运算符前加随机空格
        strcat(buf, op);   // 拼接运算符
        gen_space();       // 运算符后加随机空格
    }
}

static void gen_rand_expr() {
    size_t current_len = strlen(buf);
    // 安全防护：若剩余空间不足32字节，直接生成数字（避免递归溢出）
    if (current_len + 32 >= sizeof(buf)) {
        gen_num();
        return;
    }

    switch (choose(3)) {  // 3种生成方式随机选
    case 0:  // 1. 生成数字（递归终止）
        gen_num();
        break;

    case 1:  // 2. 生成带括号的表达式：( 表达式 )
        // 检查括号所需空间（至少" ( ) "4字节）
        if (current_len + 4 < sizeof(buf)) {
            gen_space();
            strcat(buf, "(");  // 左括号
            gen_space();
            gen_rand_expr();   // 递归生成括号内表达式
            gen_space();
            strcat(buf, ")");  // 右括号
            gen_space();
        }
        else {
            gen_num();  // 空间不足时降级生成数字
        }
        break;

    default:  // 3. 生成二元表达式：左表达式 + 运算符 + 右表达式
        gen_rand_expr();  // 左表达式
        gen_rand_op();    // 运算符
        gen_rand_expr();  // 右表达式
        break;
    }
  
}

int main(int argc, char *argv[]) {
  int seed = time(0);
  srand(seed);
  int loop = 1;
  if (argc > 1) {
    sscanf(argv[1], "%d", &loop);
  }
  int i;
  for (i = 0; i < loop; i ++) {
      memset(buf, 0, sizeof(buf));
    gen_rand_expr();

    sprintf(code_buf, code_format, buf);

    FILE *fp = fopen("/tmp/.code.c", "w");
    assert(fp != NULL);
    fputs(code_buf, fp);
    fclose(fp);

    int ret = system("gcc /tmp/.code.c -o /tmp/.expr -Wno-overflow");
    if (ret != 0) continue;

    fp = popen("/tmp/.expr", "r");
    assert(fp != NULL);

    int result;
    ret = fscanf(fp, "%d", &result);
    pclose(fp);

    printf("%u %s\n", result, buf);
  }
  return 0;
}
