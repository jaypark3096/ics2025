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

/* We use the POSIX regex functions to process regular expressions.
 * Type 'man regex' for more information about POSIX regex functions.
 */
#include <regex.h>
#include <memory/paddr.h>

enum {
    TK_NOTYPE = 256,
    TK_EQ,         // 已有的"等于"标记
    TK_NE,         // !=  新增：不等于
    TK_AND,        // &&  新增：逻辑与
    TK_PLUS,       // 加号 '+'
    TK_MINUS,      // 减号 '-'
    TK_MUL,        // 乘号 '*'
    TK_DIV,        // 除号 '/'
    TK_LPARENTHESIS, // 左括号 '('
    TK_RPARENTHESIS, // 右括号 ')'
    TK_INT,
    TK_NEG,        // - (负号，新增)
    TK_REG,        // 寄存器（$开头） 新增
    TK_DEREF,      // 指针解引用（*） 新增
};

static struct rule {
    const char* regex;
    int token_type;
} rules[] = {
  {" +", TK_NOTYPE},    // spaces
  {"&&", TK_AND},       // 逻辑与（新增）
  {"!=", TK_NE},        // 不等于（新增）
  {"\\+", TK_PLUS},     // plus
  {"-", TK_MINUS},      // minus
  {"\\*", TK_MUL},
  {"/", TK_DIV},
  {"\\(", TK_LPARENTHESIS},
  {"\\)", TK_RPARENTHESIS},
  {"0x[0-9a-fA-F]+", TK_INT}, // 十六进制整数（0x前缀）
  {"[0-9]+", TK_INT},   // 匹配一个或多个0-9之间的数字
  {"==", TK_EQ},        // equal
  {"\\$[a-zA-Z0-9]+", TK_REG}, // 寄存器（$开头，新增）
};

#define NR_REGEX ARRLEN(rules)

static regex_t re[NR_REGEX] = {};

void init_regex() {
    int i;
    char error_msg[128];
    int ret;

    for (i = 0; i < NR_REGEX; i++) {
        ret = regcomp(&re[i], rules[i].regex, REG_EXTENDED);
        if (ret != 0) {
            regerror(ret, &re[i], error_msg, 128);
            panic("regex compilation failed: %s\n%s", error_msg, rules[i].regex);
        }
    }
}

typedef struct token {
    int type;
    char str[128];
} Token;

static Token tokens[128] __attribute__((used)) = {};
static int nr_token __attribute__((used)) = 0;

static word_t eval(int p, int q, bool* success);

static bool make_token(char* e) {
    int position = 0;
    int i;
    regmatch_t pmatch;

    nr_token = 0;

    while (e[position] != '\0') {
        for (i = 0; i < NR_REGEX; i++) {
            if (regexec(&re[i], e + position, 1, &pmatch, 0) == 0 && pmatch.rm_so == 0) {
                char* substr_start = e + position;
                int substr_len = pmatch.rm_eo;

                Log("match rules[%d] = \"%s\" at position %d with len %d: %.*s",
                    i, rules[i].regex, position, substr_len, substr_len, substr_start);

                position += substr_len;
                if (rules[i].token_type == TK_NOTYPE) continue;

                assert(substr_len < 128 && nr_token < 128);
                tokens[nr_token].type = rules[i].token_type;
                strncpy(tokens[nr_token].str, substr_start, substr_len);
                tokens[nr_token].str[substr_len] = '\0';
                nr_token++;
                break;
            }
        }
        if (i == NR_REGEX) {
            printf("no match at position %d\n%s\n%*.s^\n", position, e, position, "");
            return false;
        }
    }
    return true;
}

word_t expr(char* e, bool* success) {
    *success = false;
    if (!make_token(e)) {
        return 0;
    }

    // 区分一元减号(TK_NEG)和二元减号(TK_MINUS)
    for (int i = 0; i < nr_token; i++) {
        if (tokens[i].type == TK_MINUS) {
            if (i == 0 ||  // 表达式开头
                tokens[i - 1].type == TK_LPARENTHESIS ||
                tokens[i - 1].type == TK_PLUS ||
                tokens[i - 1].type == TK_MINUS ||
                tokens[i - 1].type == TK_MUL ||
                tokens[i - 1].type == TK_DIV ||
                tokens[i - 1].type == TK_EQ ||
                tokens[i - 1].type == TK_NE ||
                tokens[i - 1].type == TK_AND) {
                tokens[i].type = TK_NEG;
            }
        }
    }

    // 区分指针解引用(TK_DEREF)和乘法(TK_MUL)
    for (int i = 0; i < nr_token; i++) {
        if (tokens[i].type == TK_MUL) {
            if (i == 0 ||
                tokens[i - 1].type == TK_LPARENTHESIS ||
                tokens[i - 1].type == TK_PLUS ||
                tokens[i - 1].type == TK_MINUS ||
                tokens[i - 1].type == TK_MUL ||
                tokens[i - 1].type == TK_DIV ||
                tokens[i - 1].type == TK_EQ ||
                tokens[i - 1].type == TK_NE ||
                tokens[i - 1].type == TK_AND) {
                tokens[i].type = TK_DEREF;
            }
        }
    }

    word_t result = eval(0, nr_token - 1, success);
    return result;
}

static bool check_parentheses(int p, int q) {
    if (tokens[p].type != TK_LPARENTHESIS || tokens[q].type != TK_RPARENTHESIS) {
        return false;
    }

    int cnt = 0;
    for (int i = p; i <= q; i++) {
        if (tokens[i].type == TK_LPARENTHESIS) cnt++;
        else if (tokens[i].type == TK_RPARENTHESIS) cnt--;

        if (cnt == 0 && i < q) return false;
        if (cnt < 0) return false;
    }

    return cnt == 0;
}

static int main_operator(int p, int q) {
    int op = -1;
    int min_priority = 10;
    int in_paren = 0;

    for (int i = p; i <= q; i++) {
        if (tokens[i].type == TK_LPARENTHESIS) {
            in_paren++;
            continue;
        }
        if (tokens[i].type == TK_RPARENTHESIS) {
            in_paren--;
            continue;
        }

        if (in_paren > 0) continue;

        int priority = -1;
        switch (tokens[i].type) {
        case TK_AND:      priority = 0; break;
        case TK_EQ:
        case TK_NE:       priority = 1; break;
        case TK_PLUS:
        case TK_MINUS:    priority = 2; break;
        case TK_MUL:
        case TK_DIV:      priority = 3; break;
        default: continue;
        }

        if (priority >= 0 && priority <= min_priority) {
            min_priority = priority;
            op = i;
        }
    }

    return op;
}

static word_t eval(int p, int q, bool* success) {
    if (p > q) {
        *success = false;
        return 0;
    }
    else if (p == q) {
        // 处理原子（寄存器或整数）
        if (tokens[p].type == TK_REG) {
            char* reg_name = tokens[p].str + 1;
            word_t reg_val = isa_reg_str2val(reg_name, success);
            if (!*success) {
                printf("Unknown register: %s\n", reg_name);
                return 0;
            }
            *success = true;
            return reg_val;
        }
        else if (tokens[p].type == TK_INT) {
            char* endptr;
            uint64_t val = strtoull(tokens[p].str, &endptr, 0);
            if (*endptr != '\0') {
                *success = false;
                return 0;
            }
            *success = true;
            return (word_t)val;
        }
        else {
            *success = false;
            return 0;
        }
    }
    // 处理括号
    else if (check_parentheses(p, q)) {
        return eval(p + 1, q - 1, success);
    }

    // 先寻找二元运算符作为主运算符（优先级最低）
    int op = main_operator(p, q);
    if (op != -1) {
        // 计算左右两边的值
        word_t val1 = eval(p, op - 1, success);
        if (!*success) return 0;

        word_t val2 = eval(op + 1, q, success);
        if (!*success) return 0;

        // 执行二元运算
        switch (tokens[op].type) {
        case TK_PLUS:  return val1 + val2;
        case TK_MINUS: return val1 - val2;
        case TK_MUL:   return val1 * val2;
        case TK_DIV:
            if (val2 == 0) {
                *success = false;
                printf("Division by zero\n");
                return 0;
            }
            return val1 / val2;
        case TK_EQ:    return (val1 == val2) ? 1 : 0;
        case TK_NE:    return (val1 != val2) ? 1 : 0;
        case TK_AND:   return (val1 && val2) ? 1 : 0;
        default:
            *success = false;
            return 0;
        }
    }

    // 若没有二元运算符，再处理一元运算符（负号或解引用）
    if (tokens[p].type == TK_NEG || tokens[p].type == TK_DEREF) {
        word_t val = eval(p + 1, q, success);
        if (!*success) return 0;

        if (tokens[p].type == TK_NEG) {
            return -val;
        }
        else { // TK_DEREF
            return paddr_read(val, 4);
        }
    }

    // 无法解析的表达式
    *success = false;
    return 0;
}
