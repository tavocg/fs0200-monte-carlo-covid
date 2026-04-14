TITLE = anteproyecto

all: $(TITLE).pdf

$(TITLE).pdf: $(TITLE)/$(TITLE).tex
	tectonic $< -o .

.PHONY: watch
watch:
	@while true; do \
		make -s $(TITLE).pdf ;\
		sleep 0.1 ;\
	done

.PHONY: clean
clean:
	rm -rf $(TITLE).pdf
