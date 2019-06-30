all: $(patsubst template/%, result/%, $(wildcard template/*.cpp))

result/%.cpp: dragonite.py template/%.cpp
	python3 $^ $@

clean:
	$(RM) result/*.cpp

.DELETE_ON_ERROR:
