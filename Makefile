all: $(patsubst template/%, result/%, $(wildcard template/*.cpp))
	-clang-format -i result/*.cpp

result/%.cpp: dragonite.py template/%.cpp
	python3 $^ $@

clean:
	$(RM) result/*.cpp
