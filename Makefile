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
  FRAMEWORKS := -framework Metal -framework Foundation -framework MetalKit
  CFLAGS += -DPLATFORM_MACOS

  # All source files (C + ObjC)
  C_SRCS  := $(shell find $(SRC_DIR) -name '*.c')
  OC_SRCS := $(shell find $(SRC_DIR) -name '*.m')
  C_OBJS  := $(patsubst $(SRC_DIR)/%.c,$(BUILD)/%.o,$(C_SRCS))
  OC_OBJS := $(patsubst $(SRC_DIR)/%.m,$(BUILD)/%.o,$(OC_SRCS))
  OBJS    := $(C_OBJS) $(OC_OBJS)
  LDFLAGS += $(FRAMEWORKS)
else
  # Linux: pure C subset only (Phase 1 testing)
  CFLAGS += -DPLATFORM_LINUX
  C_SRCS := $(shell find $(SRC_DIR) -name '*.c')
  OBJS   := $(patsubst $(SRC_DIR)/%.c,$(BUILD)/%.o,$(C_SRCS))
endif

# Targets
TARGET := ingot

.PHONY: all clean debug release

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

clean:
	rm -rf $(BUILD) $(TARGET)

# Include dependency files
-include $(shell find $(BUILD) -name '*.d' 2>/dev/null)
