cmd_/home/hook/ics2025/nemu/build/obj-riscv32-nemu-interpreter/src/device/intr.o := unused

source_/home/hook/ics2025/nemu/build/obj-riscv32-nemu-interpreter/src/device/intr.o := src/device/intr.c

deps_/home/hook/ics2025/nemu/build/obj-riscv32-nemu-interpreter/src/device/intr.o := \
  /home/hook/ics2025/nemu/include/isa.h \
  /home/hook/ics2025/nemu/src/isa/riscv32/include/isa-def.h \
    $(wildcard include/config/rve.h) \
    $(wildcard include/config/rv64.h) \
  /home/hook/ics2025/nemu/include/common.h \
    $(wildcard include/config/target/am.h) \
    $(wildcard include/config/mbase.h) \
    $(wildcard include/config/msize.h) \
    $(wildcard include/config/isa64.h) \
  /home/hook/ics2025/nemu/include/macro.h \
  /home/hook/ics2025/nemu/include/debug.h \
  /home/hook/ics2025/nemu/include/utils.h \
    $(wildcard include/config/target/native/elf.h) \

/home/hook/ics2025/nemu/build/obj-riscv32-nemu-interpreter/src/device/intr.o: $(deps_/home/hook/ics2025/nemu/build/obj-riscv32-nemu-interpreter/src/device/intr.o)

$(deps_/home/hook/ics2025/nemu/build/obj-riscv32-nemu-interpreter/src/device/intr.o):
