deps_config := \
	/home/timmo/.local/ecos-sdk/tools/kconfig/Kconfig

include/config/auto.conf: \
	$(deps_config)


$(deps_config): ;
