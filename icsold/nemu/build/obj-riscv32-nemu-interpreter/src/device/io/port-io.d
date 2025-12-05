cmd_/home/hook/ics2025/nemu/build/obj-riscv32-nemu-interpreter/src/device/io/port-io.o := unused

source_/home/hook/ics2025/nemu/build/obj-riscv32-nemu-interpreter/src/device/io/port-io.o := src/device/io/port-io.c

deps_/home/hook/ics2025/nemu/build/obj-riscv32-nemu-interpreter/src/device/io/port-io.o := \
  /home/hook/ics2025/nemu/include/device/map.h \
  /home/hook/ics2025/nemu/include/cpu/difftest.h \
    $(wildcard include/config/difftest.h) \
  /home/hook/ics2025/nemu/include/common.h \
    $(wildcard include/config/target/am.h) \
    $(wildcard include/config/mbase.h) \
    $(wildcard include/config/msize.h) \
    $(wildcard include/config/isa64.h) \
  /home/hook/ics2025/nemu/include/macro.h \
  /home/hook/ics2025/nemu/include/debug.h \
  /home/hook/ics2025/nemu/include/utils.h \
    $(wildcard include/config/target/native/elf.h) \
  /home/hook/ics2025/nemu/include/difftest-def.h \
    $(wildcard include/config/isa/x86.h) \
    $(wildcard include/config/isa/mips32.h) \
    $(wildcard include/config/isa/riscv.h) \
    $(wildcard include/config/rv64.h) \
    $(wildcard include/config/rve.h) \
    $(wildcard include/config/isa/loongarch32r.h) \

/home/hook/ics2025/nemu/build/obj-riscv32-nemu-interpreter/src/device/io/port-io.o: $(deps_/home/hook/ics2025/nemu/build/obj-riscv32-nemu-interpreter/src/device/io/port-io.o)

$(deps_/home/hook/ics2025/nemu/build/obj-riscv32-nemu-interpreter/src/device/io/port-io.o):
