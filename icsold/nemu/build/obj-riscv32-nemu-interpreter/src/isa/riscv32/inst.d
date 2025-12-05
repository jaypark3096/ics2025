cmd_/home/hook/ics2025/nemu/build/obj-riscv32-nemu-interpreter/src/isa/riscv32/inst.o := unused

source_/home/hook/ics2025/nemu/build/obj-riscv32-nemu-interpreter/src/isa/riscv32/inst.o := src/isa/riscv32/inst.c

deps_/home/hook/ics2025/nemu/build/obj-riscv32-nemu-interpreter/src/isa/riscv32/inst.o := \
  src/isa/riscv32/local-include/reg.h \
    $(wildcard include/config/rt/check.h) \
    $(wildcard include/config/rve.h) \
  /home/hook/ics2025/nemu/include/common.h \
    $(wildcard include/config/target/am.h) \
    $(wildcard include/config/mbase.h) \
    $(wildcard include/config/msize.h) \
    $(wildcard include/config/isa64.h) \
  /home/hook/ics2025/nemu/include/macro.h \
  /home/hook/ics2025/nemu/include/debug.h \
  /home/hook/ics2025/nemu/include/utils.h \
    $(wildcard include/config/target/native/elf.h) \
  /home/hook/ics2025/nemu/include/cpu/cpu.h \
  /home/hook/ics2025/nemu/include/cpu/ifetch.h \
  /home/hook/ics2025/nemu/include/memory/vaddr.h \
  /home/hook/ics2025/nemu/include/cpu/decode.h \
    $(wildcard include/config/itrace.h) \
  /home/hook/ics2025/nemu/include/isa.h \
  /home/hook/ics2025/nemu/src/isa/riscv32/include/isa-def.h \
    $(wildcard include/config/rv64.h) \

/home/hook/ics2025/nemu/build/obj-riscv32-nemu-interpreter/src/isa/riscv32/inst.o: $(deps_/home/hook/ics2025/nemu/build/obj-riscv32-nemu-interpreter/src/isa/riscv32/inst.o)

$(deps_/home/hook/ics2025/nemu/build/obj-riscv32-nemu-interpreter/src/isa/riscv32/inst.o):
