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

#include "sdb.h"
#include <stdio.h>
#include <string.h>
#define NR_WP 32



static WP wp_pool[NR_WP] = {};
static WP *head = NULL, *free_ = NULL;

void init_wp_pool() {
  int i;
  for (i = 0; i < NR_WP; i ++) {
    wp_pool[i].NO = i;
    wp_pool[i].next = (i == NR_WP - 1 ? NULL : &wp_pool[i + 1]);
    wp_pool[i].expr[0] = '\0';
  }

  head = NULL;
  free_ = wp_pool;
}

WP* new_wp(char* expr_str) {
    if (free_ == NULL) {
        printf("Error: No free watchpoints available\n");
        return NULL;
    }

    // 从空闲链表取一个节点
    WP* wp = free_;
    free_ = free_->next;

    // 初始化监视点
    strncpy(wp->expr, expr_str, sizeof(wp->expr) - 1);
    wp->expr[sizeof(wp->expr) - 1] = '\0';

    // 计算初始值
    bool success;
    wp->value = expr(expr_str, &success);
    if (!success) {
        printf("Error: Invalid expression '%s'\n", expr_str);
        // 放回空闲链表
        wp->next = free_;
        free_ = wp;
        return NULL;
    }

    // 添加到使用中链表
    wp->next = head;
    head = wp;
    return wp;
}


// 删除指定编号的监视点
bool free_wp(int no) {
    WP* prev = NULL, * curr = head;

    // 查找监视点
    while (curr != NULL) {
        if (curr->NO == no) {
            // 从使用中链表移除
            if (prev == NULL) head = curr->next;
            else prev->next = curr->next;

            // 添加到空闲链表
            curr->next = free_;
            free_ = curr;
            return true;
        }
        prev = curr;
        curr = curr->next;
    }
    printf("Error: Watchpoint %d not found\n", no);
    return false;
}

// 检查所有监视点值是否变化
bool check_wp() {
    bool changed = false;
    WP* wp = head;

    while (wp != NULL) {
        bool success;
        word_t new_val = expr(wp->expr, &success);
        if (!success) {
            printf("Error: Failed to evaluate watchpoint %d: %s\n", wp->NO, wp->expr);
            wp = wp->next;
            continue;
        }

        if (new_val != wp->value) {
            printf("Watchpoint %d triggered:\n", wp->NO);
            printf("  %s: 0x%x -> 0x%x\n", wp->expr, wp->value, new_val);
            wp->value = new_val;
            changed = true;
        }
        wp = wp->next;
    }

    return changed;
}

// 获取所有监视点（用于info w命令）
WP* get_wp_head() {
    return head;
}

