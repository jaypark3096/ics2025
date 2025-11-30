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

#ifndef __SDB_H__
#define __SDB_H__

#include <common.h>

// 完整定义 WP 结构体（包含所有成员）
typedef struct watchpoint {
	int NO;                     //  watchpoint 编号
	struct watchpoint* next;    //  链表下一个节点
	char expr[128];             //  监控的表达式
	word_t value;               //  表达式的当前值
} WP;

word_t expr(char *e, bool *success);
bool check_wp();

WP* get_wp_head();
WP* new_wp(char* expr_str);   // 新增：创建新 watchpoint
bool free_wp(int no);         // 新增：删除指定 watchpoint


#endif
