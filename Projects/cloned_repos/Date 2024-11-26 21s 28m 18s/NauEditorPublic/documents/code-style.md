# Nau Editor Coding Style Guide

## Introduction

This is Nau Editor C++ coding style guide that will be used as reference and enforced during code reviews. It's important to use a guide like this, because people think in patterns and following a strict guide makes it way easier for other developers to follow your code. We want our codebase to be as consistent as possible. Please refer to [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) for anything that is not covered here. Any suggestions for additions are welcome.

## Editor settings

Use **4 spaces** for indentation.

In Visual Studio set the following setting checked:

```
Tools -> Options -> Environment -> Documents -> Save documents as Unicode when data cannot be saved in codepage
```

## Naming conventions

Always prefer readability. Don't use short, cryptic names.

Use **PascalCase** for classes and functions:

```cpp
class Orc
{
public:
    virtual void attack(Entity& entity) = 0;
};

void OrcBattle(Orc& a, Orc& b)
{
}
```

Use **camelCase** for methods and variables:

```cpp
class OrcWizard : public Orc
{
public:
    void attack(Entity& entity) override;
    void castSpell(Entity& entity);
};

const auto fishSoup = Cooker::cook("Fish Soup");
```

Put functions inside namespace Nau:

```cpp
namespace Nau
{
    bool ShowWarningDialog();
}
```

### Files

Prefix file names with **nau_**:

```
nau_editor.cpp
nau_logger.cpp
```

Header files should use **.hpp** as an extension:

```
nau_editor.hpp
nau_logger.hpp
```

### Classes

Prefix class names with **Nau**:

```cpp
class NauEditor
{
    ...    
};
```

Non-const class members should be private and prefixed with **_m**:

```cpp
class Elf : public Entity
{
...
private:
    int m_hp;
    int m_range;
};
```

Const class members should be public and not contain any prefixes:

```cpp
class Orc : public Entity
{
public:
    const kind = "Orc";
};
```

Put methods and member variables into separate access specifiers:

```cpp
class Dog
{
public:
    void update();
    void bark();
    
public:
    const int numberOfLegs = 4;	
    
private:
    void pee();
    void digBoneOut();

private:
    pair<int, int> m_hiddenBoneLocation;
};
```

### Abstract classes and interfaces

Add **Abstract** prefix to the names of classes that are abstract:

```cpp
class NauAbstractButton
{
public:
    virtual void onClick() = 0;

private:
    bool m_state;
};
```

Add **Interface** suffix to the names of classes that serve as interfaces:

```cpp
class NauNetworkRequestInterface
{
public:
    virtual void onConnect() = 0;
    virtual void onPacket() = 0;
};
```

*Note: interfaces only have pure virtual methods, while abstract classes can also contain implemented methods or variable declarations*

### Spelling

Use American spelling instead of British:

```cpp
int color;         // Not colour
void organize();   // Not organise
bool analog;        // Not analogue
QPos center();     // Not centre
QColor gray;       // Not grey
// etc...
```

## Comments

Add a single space after a comment opening sequence:

```cpp
// My awesome comment    Good
//My horrible comment    Bad
```

Add no new lines between comments and code:

```cpp
// Good:
// Update the entities
update();

// Bad:
// Update the entities

update();
```

Pad inline comments with at least 2 spaces.

```cpp
class DodgyFairy
{
public:
    void flyTo(Entity* entity); // Starts to fly towards we target <--- BAD
    void landOn(Block* block);  // Lands on specific block <--- GOOD
}
```
    
Try your best to make code **self-documented**. Don't overdo comments:

```cpp
class NauWidget
{
public:
    ~NauWidget()  // Destructor  <--- BAD
    void detachFromParent();  // Detaches itself from the parent widget  <--- BAD 
};
```
   
But definitelly heavily comment sections that might be difficult to undestand.
Put TODO notes if something needs to be finished or fixed in future:

```cpp
void temporaryShittyCode()   // TODO: will be fixed in NED-1541
{
...
}
```

### Files

Add the following header on top of every header file:

```cpp
// <filename>
//
// Copyright © <year> N-GINN LLC. All rights reserved.
//
// Optional file description.
```

For example:

```cpp
// nau_widget.hpp
//
// Copyright © 2023 N-GINN LLC. All rights reserved.
//
// Set of Nau wrappers of base Qt classes.
// No naked Qt classes should be used outside of this file.


// ** NauWidget 

class NauWidget : public QWidget
{
    ...
};
```

### Classes

Add a comment in format of  `// ** <class-name>` before every class. Optionally, add class description after that:

```cpp
// ** NauOverlay
//
// Used to make transparent overlays on top of windows.
// Override events functions to receive parent events.
    
class NauOverlay
{
};
```

Prefer to omit *get* part from the name of getter methods:

```cpp
auto getPropertyName();  // Meh
auto proppertyName();    // Better
```

Don't leave any member variables uninitialized. Either initialize in-place or in the initalizer list:

```cpp
class NauClient
{
    NauClient(const string& name)
        : m_name(name)
    {
        m_status = Idle;
    }

private:
    string      m_name;              // Initialized in the initialized list - very good
    NauSocket*  m_socket = nullptr;  // Initialized in-place - good
    NauStatus   m_status             // Initialized in constructor - bad
    bool        m_running;           // Not initialized anywhere - very bad!
};
```

## Whitespace

Separate different kinds of includes (Qt, project, std, engine, etc.) with a single line. Also sort includes alphabetically:

```cpp
#include <QTableView>
#include <QWidget>

#include <iostream>
#include <vector>

#include <engine_core.hpp>
#include <engine_sound.hpp>
#include <engine_texture.hpp>
```

Separate methods of the same class with a single line:

```cpp
NauEditor::update()
{

}

NauEditor::close() {}
```

Separate different classes and functions with 2 lines, also put 2 lines after includes:

```cpp
#include <iostream>

#include "nau_utility.hpp"


// ** NauWidget

class NauWidget
{
};


// ** NauEditor

class NauEditor
{
};
```

Put an empty line at the end of every file:

```cpp
class Dog
{
...
};

<EOF>
```

Pointer should be a part of the type:

```cpp
int* ptr;   // Good
int *ptr    // Bad
int * ptr;  // Horrible! :)
```

Don't create multiple variables on a single line:

```cpp
// Bad
int x, y;  

// Good
int x;
int y;  
```

Use the following style for initializer lists:

```cpp
NauObject::NauObject(int x, int y, int z)
    : m_x(x)
    , m_y(y)
    , m_z(z)
```

### Brackets

Put opening brackets of functions, methods and classes on a new line, after the signrature:

```cpp
class NauWorker  // Good 
{
}; 

class NauWorker {  // Bad
};

void ApplyChanges()  // Good
{
}

void ApplyChanges() {  // Bad
}
```

Always use explicit brackets with if and else statements:

```cpp
// Good:
if (condition) {
    // Blah
}

if (condition) {
    // Blah
} else {
    // Blah
}

// Bad:
if (condition)
    doSomething();

if (condition) doSomething();
```

### Parenthesis

Parenthesis in logic should be explicit:

```cpp
if (i < 10 && j > 13)  // Bad
if ((i < 10) && (j > 13))  // Good
```

## Casting

Don't convert implicitly:

```cpp
int position = 13;
float positionPrecise = position;
```

Never use c-style casts!

```cpp
int variable = 10;
auto widget = (NauWidget*)(&variable);             // No, this will even compile sucessfully
auto widget = static_cast<NauWidget*>(&variable);  // Good, won't compile
```

## Qt specific

Wrap anything Qt related into Nau* classes.
Use modern signal/slot syntax:

```cpp
connect(editor, &NauEditor::event, button, &NauButton::handle);
```

Avoid  Qt containers and macros, use std or nau equivalents where possible. For instance:

```cpp
nullptr // instead of Q_NULLPTR
std::vector // instead of QVector
[[maybe_unused]] attribute // instead of Q_UNUSED
NAU_DEBUG(...) // Instead of qDebug
NAU_ASSERT(...) // Instead of Q_ASSERT, except for unit testing
```

Wrap any user-facing string literals into Qt tr() function:

```cpp
label->setText(tr("Add collider"))
```

Use <QShortcut> style of includes, instead of <qshortcut.h>

```cpp
label->setText(tr("Add collider"))
```

Use prefix signals with `event` and slots with `handle`:

```cpp
class NauButton
{
signals:
    void eventPressed();   // Good
    void pressed()         // Bad

slots:
    void handleHover();    // Good
    void hover();          // Bad
};
```

Hard-code colors using RGB(A) notation, not hex;

```cpp
const auto colorPill   = NauColor(255, 0, 0);
const auto colorGalss  = NauColor(0, 0, 255, 90);
const auto colorDirt   = NauColor(0x9b7653);
```


## Git

## Commit messages

Use imperative verb form (as if you are giving a command to someone) when writing commit messages:

```
Add new title bar     // Good
Adding new title bar  // Bad

Make change to the code style     // Good
Making change to the code style   // Bad
```

Don't just say `what` was done, say `why` it was done:

```
Change the initializer list order                                     // Bad
Fix initializer list warning by changing the order of initialization  // Good

Check for null pointer in engine initialization                       // Bad
Check for null pointer to prevent engine crash during initialization  // Good 
```

You can find more ideas for writing good commit messages here: https://www.gitkraken.com/learn/git/best-practices/git-commit-message

## General notes / unsorted

- Cover everything with tests.
- Reduce template usage to minimum.
- (Re-)read the [definitive guide](https://isocpp.org/wiki/faq/const-correctness) on const-correctness. 
- Prefer aggregation/composition to inheritance.
- Use #pragma once instead of include guards.
- `using namespace` is banned.
- Don't take any measures to restrict line width to a certain limit of characters (like 80). If a line gets too long on complex, just follow common sense.

### Pre/post-increment

Prefer pre-increment to post-increment by default, unless logically post-increment is required:

```cpp
++i;  // Preferred
i++;  // Only when necessary
```

### Premature optimazations

> *Premature optimization is the root of all evil"*
> Donald Knuth

Don't indulge in [premature optimizations](https://en.wikipedia.org/wiki/Program_optimization#When_to_optimize)!

- Good, readable prototype goes first
- Only optimize when necessary
- Don't optimize blindly, measure performance
