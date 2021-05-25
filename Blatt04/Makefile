ifeq (,$(shell sh -c 'cygpath --version 2> /dev/null'))
  # Unix
  pwd := $$(pwd)
  translate = $1
else
  # Windows mit MSys2/Cygwin
  pwd := $$(cygpath -m "$$(pwd)")
  translate = $(shell echo '$1' | sed 's/:/;/g')
endif

# Farben/ Textformatierung
BACKGR=`tput setaf 0`
GREEN=`tput setaf 10`
GREENB=`tput setab 10`
BOLD=`tput bold`
RESET=`tput sgr0`


all: Loesung.pdf

plots: Aufgabe9/plot.pdf Aufgabe10/plot.pdf

Aufgabe9/plot.pdf: Aufgabe9/Aufgabe9.py matplotlibrc header-matplotlib.tex 
				@echo 'Python-Plots:'
				TEXINPUTS="$(call translate,$(pwd):)" python Aufgabe9/Aufgabe9.py
				@echo

Aufgabe10/plot.pdf: Aufgabe10/Aufgabe10.py matplotlibrc header-matplotlib.tex 
				@echo 'Python-Plots:'
				TEXINPUTS="$(call translate,$(pwd):)" python Aufgabe10/Aufgabe10.py
				@echo

again:
				@make clean
				@make all

clean:
				@rm -rf build
				@echo ${GREEN}${BOLD}Removing build folder${RESET}

build:
				mkdir -p build

.DELETE_ON_ERROR:
# hier weitere Abhängigkeiten für build/Loesung.pdf deklarieren:
Loesung.pdf: Loesung.tex Aufgabe9/* Aufgabe9/plot.pdf Aufgabe10/* Aufgabe10/plot.pdf header.tex | build
				@TEXINPUTS="$(call translate,build:)" lualatex \
								--output-directory=build \
								--interaction=nonstopmode \
								--halt-on-error \
				Loesung.tex|grep -B 12 -e'no output PDF file produced' -e'LuaTeX' --color=auto
				@echo
				@BIBINPUTS=build: biber build/Loesung.bcf|grep -i -e'biber' -e'error' -e'errors' -e'warn' -e'warnings' --color=auto
				@echo
				@lualatex \
								--output-directory=build \
								--interaction=nonstopmode \
								--halt-on-error \
				Loesung.tex>/dev/null
				@lualatex \
								--output-directory=build \
								--interaction=nonstopmode \
								--halt-on-error \
				Loesung.tex|grep  -e'reference' -e'LuaTeX' --color=auto
				@mv build/Loesung.pdf Loesung.pdf
				@make clean
				@echo ${GREENB}${BACKGR}Success!${RESET}
