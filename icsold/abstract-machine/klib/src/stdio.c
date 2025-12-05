#include <am.h>
#include <klib.h>
#include <klib-macros.h>
#include <stdarg.h>

// ȡ�� putstr �궨�壬�����뺯����ͻ
#undef putstr

#if !defined(__ISA_NATIVE__) || defined(__NATIVE_USE_KLIB__)

// ��̬���������ڸ�ʽ�����
static char sprint_buf[1024];

// ��������ַ�������
int putchar(int c) {
    // ���ô����������
#if defined(__ISA_X86__)
    // x86 ʹ�ö˿�I/O
    asm volatile ("outb %0, %1" : : "a"(c), "d"(0x3f8));
#else
    // �����ܹ�ʹ���ڴ�ӳ��I/O
    * (volatile uint8_t*)(0xa00003f8) = (uint8_t)c;
#endif
    return c;
}

// ����ַ���
void putstr(const char* str) {
    while (*str) {
        putchar(*str++);
    }
}

// ����ַ���������
int puts(const char* str) {
    putstr(str);
    putchar('\n');
    return 0;
}

// ����ת�ַ�����������
static char* number(char* str, unsigned long num, int base, int size, int precision, int type) {
    char c, sign, tmp[66];
    const char* digits = "0123456789abcdefghijklmnopqrstuvwxyz";
    int i;

    if (type & 0x10) { // ��д��ĸ
        digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    }
    if (base < 2 || base > 36) {
        return 0;
    }

    c = (type & 0x01) ? '0' : ' ';
    sign = 0;

    if (type & 0x02) { // �з�����
        if ((long)num < 0) {
            sign = '-';
            num = -(long)num;
        }
        else if (type & 0x04) {
            sign = '+';
        }
        else if (type & 0x08) {
            sign = ' ';
        }
    }

    i = 0;
    if (num == 0) {
        tmp[i++] = '0';
    }
    else {
        while (num != 0) {
            tmp[i++] = digits[num % base];
            num /= base;
        }
    }

    if (i > precision) {
        precision = i;
    }
    size -= precision;
    if (!(type & (0x01 | 0x02))) {
        if (sign) {
            size--;
        }
    }

    if (!(type & 0x01) && !(type & 0x10)) {
        while (size-- > 0) {
            *str++ = ' ';
        }
    }

    if (sign) {
        *str++ = sign;
    }

    if (type & 0x01) { // �����
        if (!(type & 0x10)) {
            while (size-- > 0) {
                *str++ = c;
            }
        }
    }

    if (!(type & 0x10)) {
        while (i < precision--) {
            *str++ = '0';
        }
    }

    while (i-- > 0) {
        *str++ = tmp[i];
    }

    while (size-- > 0) {
        *str++ = ' ';
    }

    return str;
}

// ������vsprintfʵ��
int vsprintf(char* out, const char* fmt, va_list ap) {
    char* str = out;
    const char* s;
    unsigned long num;
    int i, base, len, flags, field_width, precision;

    for (; *fmt; fmt++) {
        if (*fmt != '%') {
            *str++ = *fmt;
            continue;
        }

        // ������־λ
        flags = 0;
    repeat:
        fmt++;
        switch (*fmt) {
        case '-': flags |= 0x01; goto repeat; // �����
        case '+': flags |= 0x04; goto repeat; // ��ʾ����
        case ' ': flags |= 0x08; goto repeat; // �ո����
        case '#': flags |= 0x10; goto repeat; // �����ʽ
        case '0': flags |= 0x20; goto repeat; // �����
        }

        // �����ֶο���
        field_width = -1;
        if ('0' <= *fmt && *fmt <= '9') {
            field_width = 0;
            do {
                field_width = field_width * 10 + *fmt - '0';
                fmt++;
            } while ('0' <= *fmt && *fmt <= '9');
        }
        else if (*fmt == '*') {
            fmt++;
            field_width = va_arg(ap, int);
        }

        // ��������
        precision = -1;
        if (*fmt == '.') {
            fmt++;
            if ('0' <= *fmt && *fmt <= '9') {
                precision = 0;
                do {
                    precision = precision * 10 + *fmt - '0';
                    fmt++;
                } while ('0' <= *fmt && *fmt <= '9');
            }
            else if (*fmt == '*') {
                fmt++;
                precision = va_arg(ap, int);
            }
            if (precision < 0) {
                precision = 0;
            }
        }

        // �����������η�
        if (*fmt == 'h' || *fmt == 'l') {
            fmt++; // �򻯴�������֧��long long
        }

        // ������ʽ�ַ�
        switch (*fmt) {
        case 'c':
            if (!(flags & 0x01)) {
                while (--field_width > 0) {
                    *str++ = ' ';
                }
            }
            *str++ = (unsigned char)va_arg(ap, int);
            while (--field_width > 0) {
                *str++ = ' ';
            }
            continue;

        case 's':
            s = va_arg(ap, char*);
            if (!s) {
                s = "(null)";
            }
            len = strlen(s);
            if (precision >= 0 && len > precision) {
                len = precision;
            }

            if (!(flags & 0x01)) {
                while (len < field_width--) {
                    *str++ = ' ';
                }
            }
            for (i = 0; i < len; i++) {
                *str++ = *s++;
            }
            while (len < field_width--) {
                *str++ = ' ';
            }
            continue;

        case 'p':
            if (field_width == -1) {
                field_width = 2 * sizeof(void*);
                flags |= 0x20; // �����
            }
            // 使用 uintptr_t 避免在 32 位系统上的指针转换警告
            str = number(str, (unsigned long)(uintptr_t)va_arg(ap, void*), 16, field_width, precision, flags);
            continue;

        case 'n':
            // ��ʵ��д�빦��
            continue;

        case '%':
            *str++ = '%';
            continue;

        case 'o':
            base = 8;
            break;

        case 'x':
        case 'X':
            base = 16;
            break;

        case 'd':
        case 'i':
            flags |= 0x02; // �з�����
        case 'u':
            base = 10;
            break;

        default:
            *str++ = '%';
            if (*fmt) {
                *str++ = *fmt;
            }
            else {
                fmt--;
            }
            continue;
        }

        if (flags & 0x02) { // �з�����
            num = va_arg(ap, int);
        }
        else {
            num = va_arg(ap, unsigned int);
        }

        str = number(str, num, base, field_width, precision, flags);
    }

    *str = '\0';
    return str - out;
}

// ��������sprintf�ļ򻯰�vsprintf������ԭ���룩
int vsprintf_simple(char* out, const char* fmt, va_list ap) {
    char* p = out;

    while (*fmt != '\0') {
        if (*fmt != '%') {
            *p++ = *fmt++;
            continue;
        }

        fmt++; // ���� '%'

        switch (*fmt) {
        case 'c': {
            *p++ = (char)va_arg(ap, int);
            break;
        }
        case 's': {
            char* str = va_arg(ap, char*);
            if (!str) str = "(null)";
            while (*str) {
                *p++ = *str++;
            }
            break;
        }
        case 'd': {
            int num = va_arg(ap, int);
            if (num < 0) {
                *p++ = '-';
                num = -num;
            }

            if (num == 0) {
                *p++ = '0';
                break;
            }

            char temp[32];
            int digits = 0;
            while (num > 0) {
                temp[digits++] = '0' + (num % 10);
                num /= 10;
            }

            for (int i = digits - 1; i >= 0; i--) {
                *p++ = temp[i];
            }
            break;
        }
        case 'x': {
            unsigned int num = va_arg(ap, unsigned int);
            if (num == 0) {
                *p++ = '0';
                break;
            }

            char temp[32];
            int digits = 0;
            while (num > 0) {
                int digit = num % 16;
                temp[digits++] = (digit < 10) ? '0' + digit : 'a' + (digit - 10);
                num /= 16;
            }

            for (int i = digits - 1; i >= 0; i--) {
                *p++ = temp[i];
            }
            break;
        }
        case 'p': {
            void* ptr = va_arg(ap, void*);
            *p++ = '0';
            *p++ = 'x';

            uintptr_t num = (uintptr_t)ptr;
            if (num == 0) {
                *p++ = '0';
                break;
            }

            char temp[32];
            int digits = 0;
            while (num > 0) {
                int digit = num % 16;
                temp[digits++] = (digit < 10) ? '0' + digit : 'a' + (digit - 10);
                num /= 16;
            }

            for (int i = digits - 1; i >= 0; i--) {
                *p++ = temp[i];
            }
            break;
        }
        case '%': {
            *p++ = '%';
            break;
        }
        default: {
            *p++ = '%';
            *p++ = *fmt;
            break;
        }
        }
        fmt++;
    }

    *p = '\0';
    return p - out;
}

// ������printfʵ��
int printf(const char* fmt, ...) {
    va_list args;
    int n;

    va_start(args, fmt);
    n = vsprintf(sprint_buf, fmt, args);
    va_end(args);

    putstr(sprint_buf);
    return n;
}

// ����ԭ�е�sprintfʵ�֣������ݣ�
int sprintf(char* out, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int n = vsprintf_simple(out, fmt, ap);
    va_end(ap);
    return n;
}

// ��ȫ�汾��sprintf�������������
int snprintf(char* out, size_t n, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);

    // ���nΪ0����ִ���κβ���
    if (n == 0) {
        va_end(ap);
        return 0;
    }

    // ʹ����ʱ������ȷ����ȫ
    char temp_buf[1024];
    int len = vsprintf_simple(temp_buf, fmt, ap);
    va_end(ap);

    // ���Ƶ������������ȷ��������n-1���ַ�
    if (len >= n) {
        memcpy(out, temp_buf, n - 1);
        out[n - 1] = '\0';
        return n - 1;
    }
    else {
        memcpy(out, temp_buf, len + 1); // ����null��ֹ��
        return len;
    }
}

// ��ȫ�汾��vsprintf�������������
int vsnprintf(char* out, size_t n, const char* fmt, va_list ap) {
    // ���nΪ0����ִ���κβ���
    if (n == 0) {
        return 0;
    }

    // ʹ����ʱ������ȷ����ȫ
    char temp_buf[1024];
    int len = vsprintf_simple(temp_buf, fmt, ap);

    // ���Ƶ������������ȷ��������n-1���ַ�
    if (len >= n) {
        memcpy(out, temp_buf, n - 1);
        out[n - 1] = '\0';
        return n - 1;
    }
    else {
        memcpy(out, temp_buf, len + 1); // ����null��ֹ��
        return len;
    }
}

#endif
