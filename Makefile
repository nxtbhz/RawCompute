CC      := gcc
CFLAGS  := -Wall -Wextra -Werror -pedantic -std=c99 -I./include \
           -Wshadow -Wformat=2  -Wstrict-prototypes  \
           -Wmissing-prototypes -Wnull-dereference                \
            -Wundef -Wwrite-strings             \
           -Wfloat-equal -Wcast-qual -Wpointer-arith              \
           -fstack-protector-strong -fno-common -ffast-math
LDFLAGS := -lm
VALGRIND := valgrind --leak-check=full --show-leak-kinds=all --errors-for-leak-kinds=all --error-exitcode=1

LIB_NAME  := rawcompute.a

SRC_DIR   := src
TEST_DIR  := tests
INC_DIR   := include

SRC_FILES := $(wildcard $(SRC_DIR)/r_*.c)
SRC_OBJS  := $(SRC_FILES:.c=.o)

TEST_SRCS := $(wildcard $(TEST_DIR)/test_*.c)
TEST_OBJS := $(TEST_SRCS:.c=.o)
TESTS     := $(TEST_SRCS:.c=)

HEADERS   := $(wildcard $(INC_DIR)/*.h)

.PHONY: all test valgrind clean format

all: $(LIB_NAME) test

$(LIB_NAME): $(SRC_OBJS)
	@echo "---- Building library $(LIB_NAME) ----"
	ar rcs $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

test: $(LIB_NAME) $(TESTS)
	@echo "---- Running all tests ----"
	@for exe in $(TESTS); do \
		echo "---- Executing $$exe ----"; \
		"$$exe" || exit $$?; \
	done
	@echo "---- All tests completed ----"

$(TESTS): % : %.o
	$(CC) $(CFLAGS) $< $(LIB_NAME) -o $@ $(LDFLAGS)

valgrind: test
	@echo "---- Starting Valgrind checks ----"
	@for exe in $(TESTS); do \
		echo "---- VALGRIND $$exe ----"; \
		$(VALGRIND) "$$exe" || exit $$?; \
	done

format:
	@echo "---- Formatting files with clang-format (Microsoft style) ----"
	clang-format -i --style=Microsoft $(SRC_FILES) $(TEST_SRCS) $(HEADERS)

clean:
	rm -f $(TESTS) $(TEST_OBJS) $(SRC_OBJS) $(LIB_NAME)
	@echo "---- Clean completed ----"
