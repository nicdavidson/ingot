CC      ?= clang
CFLAGS  := -Wall -Wextra -Wpedantic -std=c17 -Isrc
LDFLAGS :=

# Source directories
SRC_DIR := src
BUILD   := build

# Detect platform
UNAME := $(shell uname -s)

ifeq ($(UNAME),Darwin)
  # macOS: full build with Metal + Foundation
  OBJCFLAGS := -fobjc-arc
  FRAMEWORKS := -framework Metal -framework Foundation -framework MetalKit -framework Accelerate
  CFLAGS += -DPLATFORM_MACOS -DACCELERATE_NEW_LAPACK

  # All source files (C + ObjC)
  C_SRCS  := $(shell find $(SRC_DIR) -name '*.c')
  OC_SRCS := $(shell find $(SRC_DIR) -name '*.m')
  C_OBJS  := $(patsubst $(SRC_DIR)/%.c,$(BUILD)/%.o,$(C_SRCS))
  OC_OBJS := $(patsubst $(SRC_DIR)/%.m,$(BUILD)/%.o,$(OC_SRCS))

  # Shader embedding
  SHADER_SRC := $(BUILD)/shader_strings.c
  SHADER_OBJ := $(BUILD)/shader_strings.o

  OBJS    := $(C_OBJS) $(OC_OBJS) $(SHADER_OBJ)
  LDFLAGS += $(FRAMEWORKS)
else
  # Linux: pure C subset only (no Metal, no .m files)
  CFLAGS += -DPLATFORM_LINUX
  LDFLAGS += -lm
  C_SRCS := $(shell find $(SRC_DIR) -name '*.c' ! -path '*/compute/*')
  OBJS   := $(patsubst $(SRC_DIR)/%.c,$(BUILD)/%.o,$(C_SRCS))
endif

# Targets
TARGET := ingot

.PHONY: all clean debug release test

all: release

debug: CFLAGS += -g -O0 -DDEBUG -fsanitize=address,undefined
debug: LDFLAGS += -fsanitize=address,undefined
debug: $(TARGET)

release: CFLAGS += -O2 -DNDEBUG
release: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) $(LDFLAGS) -o $@

# C sources
$(BUILD)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

# Objective-C sources (macOS only)
$(BUILD)/%.o: $(SRC_DIR)/%.m
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(OBJCFLAGS) -MMD -MP -c $< -o $@

# Embed Metal shaders as C strings (macOS only)
$(SHADER_SRC): $(wildcard $(SRC_DIR)/compute/shaders/*.metal) tools/embed_shaders.sh
	@mkdir -p $(dir $@)
	bash tools/embed_shaders.sh > $@

$(SHADER_OBJ): $(SHADER_SRC)
	$(CC) -std=c17 -c $< -o $@

# Test targets
TEST_SRCS := $(wildcard tests/test_*.c)
TEST_BINS := $(patsubst tests/%.c,$(BUILD)/tests/%,$(TEST_SRCS))

# Objects shared by tests
TEST_OBJS := $(patsubst $(SRC_DIR)/%.c,$(BUILD)/%.o,$(shell find $(SRC_DIR)/util $(SRC_DIR)/config $(SRC_DIR)/tokenizer $(SRC_DIR)/chat $(SRC_DIR)/model $(SRC_DIR)/inference $(SRC_DIR)/server -name '*.c' 2>/dev/null))
ifeq ($(UNAME),Darwin)
  TEST_OBJS += $(OC_OBJS) $(SHADER_OBJ)
endif

test: $(TEST_BINS)
	@for t in $(TEST_BINS); do echo "--- $$t ---"; $$t || exit 1; done

$(BUILD)/tests/%: tests/%.c $(TEST_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

clean:
	rm -rf $(BUILD) $(TARGET)

# Include dependency files
-include $(shell find $(BUILD) -name '*.d' 2>/dev/null)
