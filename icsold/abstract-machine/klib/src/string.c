#include <klib.h>
#include <klib-macros.h>
#include <stdint.h>
#include <stddef.h>

#if !defined(__ISA_NATIVE__) || defined(__NATIVE_USE_KLIB__)

// 计算字符串长度（不包含终止符'\0'）
size_t strlen(const char* s) {
    size_t len = 0;
    while (s[len] != '\0') {
        len++;
    }
    return len;
}

// 复制字符串（包含终止符）
char* strcpy(char* dst, const char* src) {
    char* d = dst;
    while ((*d++ = *src++) != '\0');
    return dst;
}

// 复制最多n个字符，不足则补'\0'
char* strncpy(char* dst, const char* src, size_t n) {
    char* d = dst;
    size_t i;
    for (i = 0; i < n && src[i] != '\0'; i++) {
        d[i] = src[i];
    }
    for (; i < n; i++) {
        d[i] = '\0';
    }
    return dst;
}

// 字符串拼接（将src追加到dst末尾）
char* strcat(char* dst, const char* src) {
    char* d = dst;
    while (*d != '\0') d++;    // 找到dst的终止符
    while ((*d++ = *src++) != '\0');  // 追加src内容
    return dst;
}

// 字符串比较
int strcmp(const char* s1, const char* s2) {
    while (*s1 && *s2 && *s1 == *s2) {
        s1++;
        s2++;
    }
    return (unsigned char)*s1 - (unsigned char)*s2;
}

// 最多比较n个字符的字符串比较
int strncmp(const char* s1, const char* s2, size_t n) {
    if (n == 0) return 0;
    while (--n && *s1 && *s2 && *s1 == *s2) {
        s1++;
        s2++;
    }
    return (unsigned char)*s1 - (unsigned char)*s2;
}

// 内存初始化（将n个字节设为c）
void* memset(void* s, int c, size_t n) {
    unsigned char* p = (unsigned char*)s;
    unsigned char ch = (unsigned char)c;
    while (n--) {
        *p++ = ch;
    }
    return s;
}

// 内存移动（处理重叠区域）
void* memmove(void* dst, const void* src, size_t n) {
    unsigned char* d = (unsigned char*)dst;
    const unsigned char* s = (const unsigned char*)src;
    if (d < s) {
        // 目标在源前面，正向复制
        while (n--) {
            *d++ = *s++;
        }
    }
    else if (d > s) {
        // 目标在源后面，反向复制
        d += n;
        s += n;
        while (n--) {
            *--d = *--s;
        }
    }
    return dst;
}

// 内存复制（不处理重叠，效率更高）
void* memcpy(void* out, const void* in, size_t n) {
    unsigned char* dst = (unsigned char*)out;
    const unsigned char* src = (const unsigned char*)in;
    while (n--) {
        *dst++ = *src++;
    }
    return out;
}

// 内存比较
int memcmp(const void* s1, const void* s2, size_t n) {
    const unsigned char* p1 = (const unsigned char*)s1;
    const unsigned char* p2 = (const unsigned char*)s2;
    while (n--) {
        if (*p1 != *p2) {
            return *p1 - *p2;
        }
        p1++;
        p2++;
    }
    return 0;
}

#endif
