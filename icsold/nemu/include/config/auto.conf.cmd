deps_config := \
	src/device/Kconfig \
	src/memory/Kconfig \
	src/isa/riscv32/Kconfig \
	/home/hook/ics2025/nemu/Kconfig

include/config/auto.conf: \
	$(deps_config)


$(deps_config): ;
