#---------------------------------------------------------------------------------
# Clear the implicit built in rules
#---------------------------------------------------------------------------------
.SUFFIXES:
#---------------------------------------------------------------------------------

THIS_MAKEFILE     := $(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_DIRECTORY := $(abspath $(dir $(THIS_MAKEFILE)))

include $(CURRENT_DIRECTORY)/../pe_rules

#---------------------------------------------------------------------------------
# TARGET is the name of the output
# BUILD is the directory where object files & intermediate files will be placed
# SOURCES is a list of directories containing source code
# INCLUDES is a list of directories containing extra header files
#---------------------------------------------------------------------------------
TARGET		:=	$(notdir $(CURDIR))
BUILD		:=	build
SOURCES		:=	source
DATA		:=	data
INCLUDES	:=

#---------------------------------------------------------------------------------
# options for code generation
#---------------------------------------------------------------------------------

CFLAGS	= -O1 $(MACHDEP) $(INCLUDE)
CXXFLAGS	=	$(CFLAGS)
CPPFLAGS = $(INCLUDE)


DEFFILES := $(foreach dir,$(CURRENT_DIRECTORY)/$(SOURCES),$(wildcard $(dir)/*.def))
ifeq ($(SUBST_PATH), 1)
DEFS	:= $(foreach def,$(DEFFILES),-DEF:$(subst \,\\,$(shell $(PATH_CONVERT) $(def))))
else
DEFS	:= $(foreach def,$(DEFFILES),-DEF:$(shell $(PATH_CONVERT) $(def)))
endif
LDFLAGS	=	-DRIVER -DLL $(DEFS) -SECTION:INIT,d -NODEFAULTLIB -ALIGN:0x20 -SUBSYSTEM:NATIVE -ENTRY:HalInitSystem -BASE:0x80010000 

#---------------------------------------------------------------------------------
# any extra libraries we wish to link with the project
#---------------------------------------------------------------------------------
LIBDIR  := $(CURRENT_DIRECTORY)/../lib
LIBS	:=	$(LIBDIR)/NTOSKRNL.LIB

#---------------------------------------------------------------------------------
# no real need to edit anything past this point unless you need to add additional
# rules for different file extensions
#---------------------------------------------------------------------------------
ifneq ($(BUILD),$(notdir $(CURDIR)))
#---------------------------------------------------------------------------------

export OUTPUT	:=	$(CURDIR)/$(TARGET)

export VPATH	:=	$(foreach dir,$(SOURCES),$(CURDIR)/$(dir)) \
					$(foreach dir,$(DATA),$(CURDIR)/$(dir))

export DEPSDIR	:=	$(CURDIR)/$(BUILD)

#---------------------------------------------------------------------------------
# automatically build a list of object files for our project
#---------------------------------------------------------------------------------
CFILES		:=	$(foreach dir,$(SOURCES),$(notdir $(wildcard $(dir)/*.c)))
CPPFILES	:=	$(foreach dir,$(SOURCES),$(notdir $(wildcard $(dir)/*.cpp)))
ASMFILES	:=	$(foreach dir,$(SOURCES),$(notdir $(wildcard $(dir)/*.asm)))
SFILES		:=	$(foreach dir,$(SOURCES),$(notdir $(wildcard $(dir)/*.s)))
BINFILES	:=	$(foreach dir,$(DATA),$(notdir $(wildcard $(dir)/*.*)))
RCFILES 	:=	$(foreach dir,$(SOURCES),$(notdir $(wildcard $(dir)/*.rc)))

export OFILES_BIN	:=	$(addsuffix .obj,$(BINFILES))
export OFILES_SOURCES := $(CPPFILES:.cpp=.obj) $(CFILES:.c=.obj) $(ASMFILES:.asm=.obj) $(SFILES:.s=.obj)
export RESFILES := $(RCFILES:.rc=.res)
export OFILES := $(OFILES_BIN) $(OFILES_SOURCES)

export HFILES := $(addsuffix .h,$(subst .,_,$(BINFILES)))

export OUTPUT	:=	$(CURDIR)/$(TARGET)
.PHONY: $(BUILD) clean

#---------------------------------------------------------------------------------
$(BUILD):
	@[ -d $@ ] || mkdir -p $@
	@$(MAKE) --no-print-directory -C $(BUILD) -f $(CURDIR)/Makefile

#---------------------------------------------------------------------------------
clean:
	@echo clean ...
	@rm -fr $(BUILD) $(OUTPUT).dll $(OUTPUT).dbg


#---------------------------------------------------------------------------------
else

DEPENDS	:=	$(OFILES:.obj=.d)

#---------------------------------------------------------------------------------
# main targets
#---------------------------------------------------------------------------------
$(OUTPUT).dll: $(OFILES) $(RESFILES)

$(OFILES_SOURCES) : $(HFILES)

-include $(DEPENDS)

#---------------------------------------------------------------------------------
endif
#---------------------------------------------------------------------------------
