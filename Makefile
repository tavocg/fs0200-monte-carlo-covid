TITLE ?= anteproyecto

all: $(TITLE).pdf

$(TITLE).pdf: $(TITLE)/$(TITLE).tex
	tectonic $< -o .

.PHONY: anteproyecto
anteproyecto:
	$(MAKE) TITLE=anteproyecto

.PHONY: proyecto
proyecto: proyecto-figures
	$(MAKE) TITLE=proyecto

.PHONY: proyecto-figures
proyecto-figures:
	python proyecto/generar_graficos.py

.PHONY: watch
watch:
	@while true; do \
		make -s $(TITLE).pdf ;\
		sleep 0.1 ;\
	done

.PHONY: clean
clean:
	rm -rf $(TITLE).pdf
