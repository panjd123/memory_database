CC := g++
NVCC := nvcc

# Define Arrow path
ARROW_PATH := /path/to/arrow

# 检查 NVCC 是否可用
NVCC_AVAILABLE := $(shell which $(NVCC) > /dev/null 2>&1 && echo "yes" || echo "no")

OBJ_DIR := ./obj
SRC_DIR := ./code

# 检测 AVX512 支持
AVX512_FLAGS := -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl
check_CFLAGS = $(shell if $(CC) $(1) -E -c /dev/null > /dev/null 2>&1; then echo "$(1)"; else echo ""; fi;)
SUPPORTED_AVX512_FLAGS := $(foreach flag,$(AVX512_FLAGS),$(call check_CFLAGS,$(flag)))

# Common flags for GCC
COMMON_FLAGS := -I./include -I$(ARROW_PATH)/include  -L$(ARROW_PATH)/lib64 -g -O0 -lpthread -mcmodel=large -std=c++17 -Wmultichar -lnuma -larrow -lparquet -w
COMMON_FLAGS += $(SUPPORTED_AVX512_FLAGS)
ifneq ($(SUPPORTED_AVX512_FLAGS),)
    COMMON_FLAGS += -msse
endif

# Common flags for NVCC
NVCC_FLAGS := -I./include -g -O0 -std=c++17 -Xcompiler "-Wall -Wextra -mcmodel=large" -w -lnuma

# 定义所有模块
MODULES := select project join group aggregation starjoin OLAPcore TPCH_Q5_operator TPCH_Q5
CUDA_MODULES := GPU_OLAPcore

# 如果 NVCC 不可用，从 ALL_MODULES 中移除 CUDA_MODULES
ifeq ($(NVCC_AVAILABLE),no)
    ALL_MODULES := $(MODULES)
    $(info NVCC not found. CUDA modules will not be compiled.)
else
    ALL_MODULES := $(MODULES) $(CUDA_MODULES)
endif

# 为每个模块定义源文件、目标文件和编译标志
define MODULE_template
SRC_$(1) := $$(wildcard $$(SRC_DIR)/$(1)/*.cpp) $$(wildcard $$(SRC_DIR)/$(1)/*.cu)
OBJ_$(1) := $$(patsubst $$(SRC_DIR)/$(1)/%.cpp,$$(OBJ_DIR)/$(1)/%.o,$$(filter %.cpp,$$(SRC_$(1)))) \
            $$(patsubst $$(SRC_DIR)/$(1)/%.cu,$$(OBJ_DIR)/$(1)/%.cu.o,$$(filter %.cu,$$(SRC_$(1))))
CFLAGS_$(1) := $$(COMMON_FLAGS)
NVCCFLAGS_$(1) := $$(NVCC_FLAGS)
endef

$(foreach module,$(ALL_MODULES),$(eval $(call MODULE_template,$(module))))

# 定义所有目标
ALL_TARGETS := $(addsuffix _test,$(ALL_MODULES))

# 默认目标
.PHONY: all
all: $(ALL_TARGETS)

# 为每个模块创建编译规则
define COMPILE_RULES
$(1)_test: $$(OBJ_$(1))
	$$(if $$(filter $(1),$(CUDA_MODULES)),\
		$$(NVCC) $$^ $$(NVCCFLAGS_$(1)) -o $$@,\
		$$(CC) $$^ $$(CFLAGS_$(1)) -o $$@)

$$(OBJ_DIR)/$(1)/%.o: $$(SRC_DIR)/$(1)/%.cpp
	@mkdir -p $$(dir $$@)
	$$(CC) -c $$(CFLAGS_$(1)) $$< -o $$@

$$(OBJ_DIR)/$(1)/%.cu.o: $$(SRC_DIR)/$(1)/%.cu
	@mkdir -p $$(dir $$@)
	$$(NVCC) -c $$(NVCCFLAGS_$(1)) $$< -o $$@
endef

$(foreach module,$(ALL_MODULES),$(eval $(call COMPILE_RULES,$(module))))

# 清理规则
.PHONY: clean
clean:
	rm -rf $(OBJ_DIR) $(ALL_TARGETS)

# 显示帮助信息
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all            - Build all modules"
	@echo "  clean          - Remove all built files"
	@echo "  <module>_test  - Build specific module (e.g., select_test, project_test)"
ifeq ($(NVCC_AVAILABLE),yes)
	@echo "                   GPU_OLAPcore_test is available"
endif
	@echo "  help           - Show this help message"

# 设置默认目标
.DEFAULT_GOAL := help
