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
#include "local-include/reg.h"
#include <cpu/cpu.h>
#include <cpu/ifetch.h>
#include <cpu/decode.h>

#define R(i) gpr(i)
#define Mr vaddr_read
#define Mw vaddr_write

// ָ������ö�٣���������RISC-V�������ͣ�
enum {
    TYPE_I, TYPE_U, TYPE_S, TYPE_J, TYPE_R, TYPE_B,
    TYPE_N, // none
};

// �Ĵ���������
#define src1R() do { *src1 = R(rs1); } while (0)
#define src2R() do { *src2 = R(rs2); } while (0)

// �����������꣨��RISC-V�淶ʵ�֣�
#define immI() do { *imm = SEXT(BITS(i, 31, 20), 12); } while(0) // I����������12λ������չ��
#define immU() do { *imm = SEXT(BITS(i, 31, 12), 20) << 12; } while(0) // U����������20λ������չ����12λ��
#define immS() do { *imm = (SEXT(BITS(i, 31, 25), 7) << 5) | BITS(i, 11, 7); } while(0) // S����������12λ������չ��
#define immJ() do { \
  *imm = SEXT( \
    (BITS(i, 31, 31) << 20) | \
    (BITS(i, 19, 12) << 12) | \
    (BITS(i, 20, 20) << 11) | \
    (BITS(i, 30, 21) << 1), \
  21); \
} while(0) // J����������21λ������չ��
#define immB() do { \
  *imm = SEXT( \
    (BITS(i, 31, 31) << 12) | \
    (BITS(i, 7, 7) << 11) | \
    (BITS(i, 30, 25) << 5) | \
    (BITS(i, 11, 8) << 1), \
  13); \
} while(0) // B����������13λ������չ��

// ���������뺯��������ָ�����ͽ����Ĵ�������������
static void decode_operand(Decode* s, int* rd, word_t* src1, word_t* src2, word_t* imm, int type) {
    uint32_t i = s->isa.inst;
    int rs1 = BITS(i, 19, 15); // Դ�Ĵ���1��rs1��
    int rs2 = BITS(i, 24, 20); // Դ�Ĵ���2��rs2��
    *rd = BITS(i, 11, 7);  // Ŀ��Ĵ�����rd��

    switch (type) {
    case TYPE_I: src1R();          immI(); break; // I�ͣ���rs1������I������
    case TYPE_U:                   immU(); break; // U�ͣ�����U������
    case TYPE_S: src1R(); src2R(); immS(); break; // S�ͣ���rs1/rs2������S������
    case TYPE_J:                   immJ(); break; // J�ͣ�����J������
    case TYPE_R: src1R(); src2R(); break;         // R�ͣ���rs1/rs2������������
    case TYPE_B: src1R(); src2R(); immB(); break; // B�ͣ���rs1/rs2������B������
    case TYPE_N: break;
    default: panic("unsupported type = %d", type);
    }
}

static int decode_exec(Decode* s) {
    s->dnpc = s->snpc; // Ĭ����һ��ָ���ַΪ��ǰPC+4
#define INSTPAT_INST(s) ((s)->isa.inst)
#define INSTPAT_MATCH(s, name, type, ... /* ִ���߼� */ ) { \
  int rd = 0; \
  word_t src1 = 0, src2 = 0, imm = 0; \
  decode_operand(s, &rd, &src1, &src2, &imm, concat(TYPE_, type)); \
  __VA_ARGS__ ; \
}

    INSTPAT_START();

    // -------------------------- R��ָ��Ĵ���-�Ĵ��������� --------------------------
    // add: �Ĵ����ӷ���x[rd] = x[rs1] + x[rs2]��
    INSTPAT("0000000 ????? ????? 000 ????? 01100 11", add, R, R(rd) = src1 + src2);
    // sub: �Ĵ���������x[rd] = x[rs1] - x[rs2]��
    INSTPAT("0100000 ????? ????? 000 ????? 01100 11", sub, R, R(rd) = src1 - src2);
    // slt: �з���С�ڱȽϣ�x[rd] = (x[rs1] <s x[rs2]) ? 1 : 0��
    INSTPAT("0000000 ????? ????? 010 ????? 01100 11", slt, R, R(rd) = ((int32_t)src1 < (int32_t)src2) ? 1 : 0);
    // sltu: �޷���С�ڱȽϣ�x[rd] = (x[rs1] <u x[rs2]) ? 1 : 0��
    INSTPAT("0000000 ????? ????? 011 ????? 01100 11", sltu, R, R(rd) = (src1 < src2) ? 1 : 0);
    // xor: ��λ���x[rd] = x[rs1] ^ x[rs2]��
    INSTPAT("0000000 ????? ????? 100 ????? 01100 11", xor, R, R(rd) = src1 ^ src2);
    // srl: �߼����ƣ�x[rd] = x[rs1] >> (x[rs2] & 0x1f)��
    INSTPAT("0000000 ????? ????? 101 ????? 01100 11", srl, R, R(rd) = src1 >> (src2 & 0x1f));
    // sra: �������ƣ�x[rd] = (x[rs1] >>s (x[rs2] & 0x1f))��
    INSTPAT("0100000 ????? ????? 101 ????? 01100 11", sra, R, R(rd) = (int32_t)src1 >> (src2 & 0x1f));
    // or: ��λ��x[rd] = x[rs1] | x[rs2]��
    INSTPAT("0000000 ????? ????? 110 ????? 01100 11", or , R, R(rd) = src1 | src2);
    // and: ��λ�루x[rd] = x[rs1] & x[rs2]��
    INSTPAT("0000000 ????? ????? 111 ????? 01100 11", and, R, R(rd) = src1 & src2);
    // sll: �߼����ƣ�x[rd] = x[rs1] << (x[rs2] & 0x1f)��
    INSTPAT("0000000 ????? ????? 001 ????? 01100 11", sll, R, R(rd) = src1 << (src2 & 0x1f));

    // RV32M �˳���ָ��
    // mul: �˷���x[rd] = x[rs1] * x[rs2]��
    INSTPAT("0000001 ????? ????? 000 ????? 01100 11", mul, R, R(rd) = src1 * src2);
    // mulh: �з��Ÿ�λ�˷���x[rd] = (x[rs1] *s x[rs2]) >> 32��
    INSTPAT("0000001 ????? ????? 001 ????? 01100 11", mulh, R, R(rd) = ((int64_t)(int32_t)src1 * (int64_t)(int32_t)src2) >> 32);
    // div: �з��ų�����x[rd] = x[rs1] /s x[rs2]��
    INSTPAT("0000001 ????? ????? 100 ????? 01100 11", div_riscv32, R, R(rd) = (int32_t)src1 / (int32_t)src2);
    // divu: �޷��ų�����x[rd] = x[rs1] /u x[rs2]��
    INSTPAT("0000001 ????? ????? 101 ????? 01100 11", divu_riscv32, R, R(rd) = src1 / src2);
    // rem: �з���������x[rd] = x[rs1] %s x[rs2]��
    INSTPAT("0000001 ????? ????? 110 ????? 01100 11", rem, R, R(rd) = (int32_t)src1 % (int32_t)src2);
    // remu: �޷���������x[rd] = x[rs1] %u x[rs2]��
    INSTPAT("0000001 ????? ????? 111 ????? 01100 11", remu, R, R(rd) = src1 % src2);

    // -------------------------- I��ָ������������� --------------------------
    // addi: �������ӷ���֧��li/mvαָ�x[rd] = x[rs1] + imm��
    INSTPAT("??????? ????? ????? 000 ????? 00100 11", addi, I, R(rd) = src1 + imm);
    // slti: 有符号小于立即数比较，x[rd] = (x[rs1] <s imm) ? 1 : 0
    INSTPAT("??????? ????? ????? 010 ????? 00100 11", slti, I, R(rd) = ((int32_t)src1 < (int32_t)imm) ? 1 : 0);
    // sltiu: �޷���С����������֧��seqzαָ�x[rd] = (x[rs1] <u imm) ? 1 : 0��
    INSTPAT("??????? ????? ????? 011 ????? 00100 11", sltiu, I, R(rd) = (src1 < imm) ? 1 : 0);
    // xori: ���������֧��notαָ�x[rd] = x[rs1] ^ imm��
    INSTPAT("??????? ????? ????? 100 ????? 00100 11", xori, I, R(rd) = src1 ^ imm);
    // ori: ��������x[rd] = x[rs1] | imm��
    INSTPAT("??????? ????? ????? 110 ????? 00100 11", ori, I, R(rd) = src1 | imm);
    // andi: �������루x[rd] = x[rs1] & imm��
    INSTPAT("??????? ????? ????? 111 ????? 00100 11", andi, I, R(rd) = src1 & imm);
    // slli: �������߼����ƣ�x[rd] = x[rs1] << (imm & 0x1f)��
    INSTPAT("0000000 ????? ????? 001 ????? 00100 11", slli, I, R(rd) = src1 << (imm & 0x1f));
    // srli: �������߼����ƣ�x[rd] = x[rs1] >> (imm & 0x1f)��
    INSTPAT("0000000 ????? ????? 101 ????? 00100 11", srli, I, R(rd) = src1 >> (imm & 0x1f));
    // srai: �������������ƣ�x[rd] = (x[rs1] >>s (imm & 0x1f))��
    INSTPAT("0100000 ????? ????? 101 ????? 00100 11", srai, I, R(rd) = (int32_t)src1 >> (imm & 0x1f));

    // I��LOADָ��ڴ����
    // lb: �����ֽڣ�������չ��x[rd] = sext(M[rs1+imm][7:0])��
    INSTPAT("??????? ????? ????? 000 ????? 00000 11", lb, I, R(rd) = SEXT(Mr(src1 + imm, 1), 8));
    // lh: ���ذ��֣�������չ��x[rd] = sext(M[rs1+imm][15:0])��
    INSTPAT("??????? ????? ????? 001 ????? 00000 11", lh, I, R(rd) = SEXT(Mr(src1 + imm, 2), 16));
    // lw: �����֣�x[rd] = M[rs1+imm][31:0]��
    INSTPAT("??????? ????? ????? 010 ????? 00000 11", lw, I, R(rd) = Mr(src1 + imm, 4));
    // lbu: �����޷����ֽڣ�����չ��x[rd] = M[rs1+imm][7:0]��
    INSTPAT("??????? ????? ????? 100 ????? 00000 11", lbu, I, R(rd) = Mr(src1 + imm, 1));
    // lhu: �����޷��Ű��֣�����չ��x[rd] = M[rs1+imm][15:0]��
    INSTPAT("??????? ????? ????? 101 ????? 00000 11", lhu, I, R(rd) = Mr(src1 + imm, 2));

    // jalr: �Ĵ��������ת��֧��retαָ�x[rd] = pc+4; pc = (rs1+imm) & ~1��
    INSTPAT("??????? ????? ????? 000 ????? 11001 11", jalr, I, R(rd) = s->pc + 4; s->dnpc = (src1 + imm) & ~1);

    // -------------------------- S��ָ��ڴ�д�� --------------------------
    // sb: �洢�ֽڣ�M[rs1+imm][7:0] = x[rs2][7:0]��
    INSTPAT("??????? ????? ????? 000 ????? 01000 11", sb, S, Mw(src1 + imm, 1, src2 & 0xff));
    // sh: �洢���֣�M[rs1+imm][15:0] = x[rs2][15:0]��
    INSTPAT("??????? ????? ????? 001 ????? 01000 11", sh, S, Mw(src1 + imm, 2, src2 & 0xffff));
    // sw: �洢�֣�M[rs1+imm][31:0] = x[rs2][31:0]��
    INSTPAT("??????? ????? ????? 010 ????? 01000 11", sw, S, Mw(src1 + imm, 4, src2));

    // -------------------------- U��ָ�� --------------------------
    // auipc: PC����������x[rd] = pc + (imm << 12)��
    INSTPAT("??????? ????? ????? ??? ????? 00101 11", auipc, U, R(rd) = s->pc + imm);
    // lui: ���ظ�λ��������x[rd] = imm << 12��
    INSTPAT("??????? ????? ????? ??? ????? 01101 11", lui, U, R(rd) = imm);

    // -------------------------- J��ָ�� --------------------------
    // jal: ��ת�����ӣ�֧��jαָ�x[rd] = pc+4; pc = pc + imm��
    INSTPAT("??????? ????? ????? ??? ????? 11011 11", jal, J, R(rd) = s->pc + 4; s->dnpc = s->pc + imm);

    // -------------------------- B��ָ�������֧�� --------------------------
    // beq: ��ȷ�֧��֧��beqz����rs1==rs2��pc += imm��
    INSTPAT("??????? ????? ????? 000 ????? 11000 11", beq, B, if (src1 == src2) s->dnpc = s->pc + imm);
    // bne: ���ȷ�֧��֧��bnez����rs1!=rs2��pc += imm��
    INSTPAT("??????? ????? ????? 001 ????? 11000 11", bne, B, if (src1 != src2) s->dnpc = s->pc + imm);
    // blt: �з���С�ڷ�֧����rs1 <s rs2��pc += imm��
    INSTPAT("??????? ????? ????? 100 ????? 11000 11", blt, B, if ((int32_t)src1 < (int32_t)src2) s->dnpc = s->pc + imm);
    // bge: �з��Ŵ��ڵ��ڷ�֧����rs1 >=s rs2��pc += imm��
    INSTPAT("??????? ????? ????? 101 ????? 11000 11", bge, B, if ((int32_t)src1 >= (int32_t)src2) s->dnpc = s->pc + imm);
    // bltu: �޷���С�ڷ�֧����rs1 <u rs2��pc += imm��
    INSTPAT("??????? ????? ????? 110 ????? 11000 11", bltu, B, if (src1 < src2) s->dnpc = s->pc + imm);
    // bgeu: �޷��Ŵ��ڵ��ڷ�֧����rs1 >=u rs2��pc += imm��
    INSTPAT("??????? ????? ????? 111 ????? 11000 11", bgeu, B, if (src1 >= src2) s->dnpc = s->pc + imm);

    // -------------------------- ����ָ�� --------------------------
    // ebreak: �ϵ�ָ��������壩
    INSTPAT("0000000 00001 00000 000 00000 11100 11", ebreak, N, NEMUTRAP(s->pc, R(10))); // R(10)Ϊa0
    // δ����ָ������쳣��
    INSTPAT("??????? ????? ????? ??? ????? ????? ??", inv, N, INV(s->pc));

    INSTPAT_END();

    R(0) = 0; // ǿ��x0Ϊ0������RISC-V�淶��
    return 0;
}

int isa_exec_once(Decode* s) {
    s->isa.inst = inst_fetch(&s->snpc, 4); // ȡָ��
    return decode_exec(s);
}