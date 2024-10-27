use serde::Deserialize;
use serde::Serialize;

use fennec_interner::StringIdentifier;
use fennec_span::Span;

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct Document {
    pub elements: Vec<Element>,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum Element {
    Text(Text),
    Code(Code),
    Tag(Tag),
    Line(Span),
    Annotation(Annotation),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct Text {
    pub span: Span,
    pub segments: Vec<TextSegment>,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct Code {
    pub span: Span,
    pub directives: Vec<StringIdentifier>,
    pub content: StringIdentifier,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum TextSegment {
    Paragraph { span: Span, content: StringIdentifier },
    InlineCode(Code),
    InlineTag(Tag),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct Annotation {
    pub span: Span,
    pub name: StringIdentifier,
    pub arguments: Option<StringIdentifier>,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct Tag {
    pub span: Span,
    pub name: StringIdentifier,
    pub kind: TagKind,
    pub description: StringIdentifier,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[non_exhaustive]
pub enum TagKind {
    Abstract,
    Access,
    Author,
    Category,
    Copyright,
    Deprecated,
    Example,
    Final,
    FileSource,
    Global,
    Ignore,
    Internal,
    License,
    Link,
    Method,
    Mixin,
    Name,
    Package,
    Param,
    Property,
    PropertyRead,
    PropertyWrite,
    SealProperties,
    NoSealProperties,
    SealMethods,
    NoSealMethods,
    ReadOnly,
    NoNamedArguments,
    Template,
    PsalmTemplate,
    Api,
    PsalmApi,
    PsalmInheritors,
    Return,
    See,
    Since,
    Static,
    StaticVar,
    SubPackage,
    Todo,
    Tutorial,
    Uses,
    Var,
    Throws,
    Version,
    ParamLaterInvokedCallable,
    ParamImmediatelyInvokedCallable,
    ParamClosureThis,
    TemplateCovariant,
    TemplateContravariant,
    Extends,
    Implements,
    Use,
    NotDeprecated,
    PhpstanImpure,
    PhpstanPure,
    Pure,
    Immutable,
    InheritDoc,
    ParamOut,
    PsalmConsistentConstructor,
    PsalmConsistentTemplates,
    PsalmParamOut,
    PsalmVar,
    PsalmParam,
    PsalmReturn,
    PsalmProperty,
    PsalmPropertyRead,
    PsalmPropertyWrite,
    PsalmMethod,
    PsalmIgnoreVar,
    PsalmSuppress,
    PsalmAssert,
    PsalmAssertIfTrue,
    PsalmAssertIfFalse,
    PsalmIfThisIs,
    PsalmThisOut,
    PsalmIgnoreNullableReturn,
    PsalmIgnoreFalsableReturn,
    PsalmSealProperties,
    PsalmNoSealProperties,
    PsalmSealMethods,
    PsalmNoSealMethods,
    PsalmInternal,
    PsalmReadOnly,
    PsalmMutationFree,
    PsalmExternalMutationFree,
    PsalmImmutable,
    PsalmPure,
    PsalmAllowPrivateMutation,
    PsalmReadOnlyAllowPrivateMutation,
    PsalmTrace,
    PsalmCheckType,
    PsalmCheckTypeExact,
    PsalmTaintSource,
    PsalmTaintSink,
    PsalmTaintEscape,
    PsalmTaintUnescape,
    PsalmTaintSpecialize,
    PsalmFlow,
    PsalmType,
    PsalmImportType,
    PsalmRequireExtends,
    PsalmRequireImplements,
    PsalmIgnoreVariableProperty,
    PsalmIgnoreVariableMethod,
    PsalmYield,
    PhpstanAssert,
    PhpstanAssertIfTrue,
    PhpstanAssertIfFalse,
    PhpstanSelfOut,
    PhpstanThisOut,
    PhpstanRequireExtends,
    PhpstanRequireImplements,
    PhpstanTemplate,
    PhpstanTemplateCovariant,
    PhpstanTemplateContravariant,
    PhpstanParam,
    PhpstanReturn,
    PhpstanVar,
    PhpstanReadOnly,
    PhpstanImmutable,
    Other,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum TagVendor {
    Phpstan,
    Psalm,
}

impl TagKind {
    /// Returns the vendor of the tag, if it has one.
    ///
    /// If the tag does not have a vendor, `None` is returned.
    pub fn get_vendor(&self) -> Option<TagVendor> {
        match self {
            Self::PsalmConsistentConstructor
            | Self::PsalmConsistentTemplates
            | Self::PsalmParamOut
            | Self::PsalmVar
            | Self::PsalmParam
            | Self::PsalmReturn
            | Self::PsalmProperty
            | Self::PsalmPropertyRead
            | Self::PsalmPropertyWrite
            | Self::PsalmMethod
            | Self::PsalmIgnoreVar
            | Self::PsalmSuppress
            | Self::PsalmAssert
            | Self::PsalmAssertIfTrue
            | Self::PsalmAssertIfFalse
            | Self::PsalmIfThisIs
            | Self::PsalmThisOut
            | Self::PsalmIgnoreNullableReturn
            | Self::PsalmIgnoreFalsableReturn
            | Self::PsalmSealProperties
            | Self::PsalmNoSealProperties
            | Self::PsalmSealMethods
            | Self::PsalmNoSealMethods
            | Self::PsalmInternal
            | Self::PsalmReadOnly
            | Self::PsalmMutationFree
            | Self::PsalmExternalMutationFree
            | Self::PsalmImmutable
            | Self::PsalmPure
            | Self::PsalmAllowPrivateMutation
            | Self::PsalmReadOnlyAllowPrivateMutation
            | Self::PsalmTrace
            | Self::PsalmCheckType
            | Self::PsalmCheckTypeExact
            | Self::PsalmTaintSource
            | Self::PsalmTaintSink
            | Self::PsalmTaintEscape
            | Self::PsalmTaintUnescape
            | Self::PsalmTaintSpecialize
            | Self::PsalmFlow
            | Self::PsalmType
            | Self::PsalmImportType
            | Self::PsalmRequireExtends
            | Self::PsalmRequireImplements
            | Self::PsalmIgnoreVariableProperty
            | Self::PsalmIgnoreVariableMethod
            | Self::PsalmYield => Some(TagVendor::Psalm),
            Self::PhpstanAssert
            | Self::PhpstanAssertIfTrue
            | Self::PhpstanAssertIfFalse
            | Self::PhpstanSelfOut
            | Self::PhpstanThisOut
            | Self::PhpstanRequireExtends
            | Self::PhpstanRequireImplements
            | Self::PhpstanTemplate
            | Self::PhpstanTemplateCovariant
            | Self::PhpstanTemplateContravariant
            | Self::PhpstanParam
            | Self::PhpstanReturn
            | Self::PhpstanVar
            | Self::PhpstanReadOnly
            | Self::PhpstanImmutable => Some(TagVendor::Phpstan),
            _ => None,
        }
    }

    /// Returns the non-vendored variant of the tag, if it exists.
    ///
    /// Note that not all vendored tags have a non-vendored variant.
    ///
    /// If the tag is not vendored, or if it does not have a non-vendored variant,
    ///  `None` is returned.
    pub fn get_non_vendored_variant(&self) -> Option<TagKind> {
        match self {
            Self::PsalmParamOut => Some(Self::ParamOut),
            Self::PsalmVar => Some(Self::Var),
            Self::PsalmParam => Some(Self::Param),
            Self::PsalmReturn => Some(Self::Return),
            Self::PsalmProperty => Some(Self::Property),
            Self::PsalmPropertyRead => Some(Self::PropertyRead),
            Self::PsalmPropertyWrite => Some(Self::PropertyWrite),
            Self::PsalmMethod => Some(Self::Method),
            Self::PsalmSealProperties => Some(Self::SealProperties),
            Self::PsalmNoSealProperties => Some(Self::NoSealProperties),
            Self::PsalmSealMethods => Some(Self::SealMethods),
            Self::PsalmNoSealMethods => Some(Self::NoSealMethods),
            Self::PsalmInternal => Some(Self::Internal),
            Self::PsalmReadOnly => Some(Self::ReadOnly),
            Self::PsalmImmutable => Some(Self::Immutable),
            Self::PsalmPure => Some(Self::Pure),
            Self::PhpstanTemplate => Some(Self::Template),
            Self::PhpstanTemplateCovariant => Some(Self::TemplateCovariant),
            Self::PhpstanTemplateContravariant => Some(Self::TemplateContravariant),
            Self::PhpstanParam => Some(Self::Param),
            Self::PhpstanReturn => Some(Self::Return),
            Self::PhpstanVar => Some(Self::Var),
            Self::PhpstanReadOnly => Some(Self::ReadOnly),
            Self::PhpstanImmutable => Some(Self::Immutable),
            _ => None,
        }
    }

    pub fn is_repeatable(&self) -> bool {
        match self {
            Self::Author
            | Self::Deprecated
            | Self::Example
            | Self::Ignore
            | Self::Link
            | Self::Method
            | Self::Mixin
            | Self::Package
            | Self::Param
            | Self::Property
            | Self::PropertyRead
            | Self::PropertyWrite
            | Self::Return
            | Self::See
            | Self::Since
            | Self::Throws
            | Self::Uses
            | Self::Var => true,
            _ => false,
        }
    }
}

impl<T> From<T> for TagKind
where
    T: AsRef<str>,
{
    fn from(value: T) -> Self {
        match value.as_ref().to_ascii_lowercase().as_str() {
            "abstract" => TagKind::Abstract,
            "access" => TagKind::Access,
            "author" => TagKind::Author,
            "category" => TagKind::Category,
            "copyright" => TagKind::Copyright,
            "deprecated" => TagKind::Deprecated,
            "example" => TagKind::Example,
            "final" => TagKind::Final,
            "filesource" => TagKind::FileSource,
            "global" => TagKind::Global,
            "ignore" => TagKind::Ignore,
            "internal" => TagKind::Internal,
            "license" => TagKind::License,
            "link" => TagKind::Link,
            "method" => TagKind::Method,
            "mixin" => TagKind::Mixin,
            "name" => TagKind::Name,
            "package" => TagKind::Package,
            "param" => TagKind::Param,
            "property" => TagKind::Property,
            "property-read" => TagKind::PropertyRead,
            "propertyread" => TagKind::PropertyRead,
            "property-write" => TagKind::PropertyWrite,
            "propertywrite" => TagKind::PropertyWrite,
            "sealproperties" => TagKind::SealProperties,
            "seal-properties" => TagKind::SealProperties,
            "nosealproperties" => TagKind::NoSealProperties,
            "no-seal-properties" => TagKind::NoSealProperties,
            "sealmethods" => TagKind::SealMethods,
            "seal-methods" => TagKind::SealMethods,
            "nosealmethods" => TagKind::NoSealMethods,
            "no-seal-methods" => TagKind::NoSealMethods,
            "readonly" => TagKind::ReadOnly,
            "nonamedarguments" => TagKind::NoNamedArguments,
            "no-named-arguments" => TagKind::NoNamedArguments,
            "template" => TagKind::Template,
            "psalm-template" => TagKind::PsalmTemplate,
            "api" => TagKind::Api,
            "psalm-api" => TagKind::PsalmApi,
            "psalm-inheritors" => TagKind::PsalmInheritors,
            "return" => TagKind::Return,
            "see" => TagKind::See,
            "since" => TagKind::Since,
            "static" => TagKind::Static,
            "staticvar" => TagKind::StaticVar,
            "static-var" => TagKind::StaticVar,
            "subpackage" => TagKind::SubPackage,
            "sub-package" => TagKind::SubPackage,
            "todo" => TagKind::Todo,
            "tutorial" => TagKind::Tutorial,
            "uses" => TagKind::Uses,
            "var" => TagKind::Var,
            "throws" => TagKind::Throws,
            "version" => TagKind::Version,
            "param-later-invoked-callable" => TagKind::ParamLaterInvokedCallable,
            "paramlaterinvokedcallable" => TagKind::ParamLaterInvokedCallable,
            "param-immediately-invoked-callable" => TagKind::ParamImmediatelyInvokedCallable,
            "paramimmediatelyinvokedcallable" => TagKind::ParamImmediatelyInvokedCallable,
            "param-closure-this" => TagKind::ParamClosureThis,
            "paramclosurethis" => TagKind::ParamClosureThis,
            "template-covariant" => TagKind::TemplateCovariant,
            "templatecovariant" => TagKind::TemplateCovariant,
            "template-contravariant" => TagKind::TemplateContravariant,
            "templatecontravariant" => TagKind::TemplateContravariant,
            "extends" => TagKind::Extends,
            "implements" => TagKind::Implements,
            "use" => TagKind::Use,
            "not-deprecated" => TagKind::NotDeprecated,
            "notdeprecated" => TagKind::NotDeprecated,
            "phpstan-impure" => TagKind::PhpstanImpure,
            "phpstan-pure" => TagKind::PhpstanPure,
            "pure" => TagKind::Pure,
            "immutable" => TagKind::Immutable,
            "inheritdoc" => TagKind::InheritDoc,
            "inherit-doc" => TagKind::InheritDoc,
            "param-out" => TagKind::ParamOut,
            "psalm-param-out" => TagKind::PsalmParamOut,
            "psalmconsistentconstructor" => TagKind::PsalmConsistentConstructor,
            "psalm-consistent-constructor" => TagKind::PsalmConsistentConstructor,
            "psalmconsistenttemplates" => TagKind::PsalmConsistentTemplates,
            "psalm-consistent-templates" => TagKind::PsalmConsistentTemplates,
            "psalm-var" => TagKind::PsalmVar,
            "psalm-param" => TagKind::PsalmParam,
            "psalm-return" => TagKind::PsalmReturn,
            "psalm-property" => TagKind::PsalmProperty,
            "psalm-property-read" => TagKind::PsalmPropertyRead,
            "psalm-propertyread" => TagKind::PsalmPropertyRead,
            "psalm-property-write" => TagKind::PsalmPropertyWrite,
            "psalm-propertywrite" => TagKind::PsalmPropertyWrite,
            "psalm-method" => TagKind::PsalmMethod,
            "psalm-ignore-var" => TagKind::PsalmIgnoreVar,
            "psalmignorevar" => TagKind::PsalmIgnoreVar,
            "psalm-suppress" => TagKind::PsalmSuppress,
            "psalm-assert" => TagKind::PsalmAssert,
            "psalm-assert-if-true" => TagKind::PsalmAssertIfTrue,
            "psalm-assertiftrue" => TagKind::PsalmAssertIfTrue,
            "psalm-assert-if-false" => TagKind::PsalmAssertIfFalse,
            "psalm-assertiffalse" => TagKind::PsalmAssertIfFalse,
            "psalm-if-this-is" => TagKind::PsalmIfThisIs,
            "psalmifthisis" => TagKind::PsalmIfThisIs,
            "psalm-this-out" => TagKind::PsalmThisOut,
            "psalmthisout" => TagKind::PsalmThisOut,
            "psalm-ignore-nullable-return" => TagKind::PsalmIgnoreNullableReturn,
            "psalmignorenullablereturn" => TagKind::PsalmIgnoreNullableReturn,
            "psalm-ignore-falsable-return" => TagKind::PsalmIgnoreFalsableReturn,
            "psalmignorefalsablereturn" => TagKind::PsalmIgnoreFalsableReturn,
            "psalm-seal-properties" => TagKind::PsalmSealProperties,
            "psalmsealproperties" => TagKind::PsalmSealProperties,
            "psalm-no-seal-properties" => TagKind::PsalmNoSealProperties,
            "psalmnosealproperties" => TagKind::PsalmNoSealProperties,
            "psalm-seal-methods" => TagKind::PsalmSealMethods,
            "psalmsealmethods" => TagKind::PsalmSealMethods,
            "psalm-no-seal-methods" => TagKind::PsalmNoSealMethods,
            "psalmnosealmethods" => TagKind::PsalmNoSealMethods,
            "psalm-internal" => TagKind::PsalmInternal,
            "psalm-readonly" => TagKind::PsalmReadOnly,
            "psalm-mutation-free" => TagKind::PsalmMutationFree,
            "psalmmutationfree" => TagKind::PsalmMutationFree,
            "psalm-external-mutation-free" => TagKind::PsalmExternalMutationFree,
            "psalmexternalmutationfree" => TagKind::PsalmExternalMutationFree,
            "psalm-immutable" => TagKind::PsalmImmutable,
            "psalm-pure" => TagKind::PsalmPure,
            "psalm-allow-private-mutation" => TagKind::PsalmAllowPrivateMutation,
            "psalmallowprivatemutation" => TagKind::PsalmAllowPrivateMutation,
            "psalm-readonly-allow-private-mutation" => TagKind::PsalmReadOnlyAllowPrivateMutation,
            "psalmreadonlyallowprivatemutation" => TagKind::PsalmReadOnlyAllowPrivateMutation,
            "psalm-trace" => TagKind::PsalmTrace,
            "psalm-check-type" => TagKind::PsalmCheckType,
            "psalmchecktype" => TagKind::PsalmCheckType,
            "psalm-check-type-exact" => TagKind::PsalmCheckTypeExact,
            "psalmchecktypeexact" => TagKind::PsalmCheckTypeExact,
            "psalm-taint-source" => TagKind::PsalmTaintSource,
            "psalmtaintsource" => TagKind::PsalmTaintSource,
            "psalm-taint-sink" => TagKind::PsalmTaintSink,
            "psalmtaintsink" => TagKind::PsalmTaintSink,
            "psalm-taint-escape" => TagKind::PsalmTaintEscape,
            "psalmtaintescape" => TagKind::PsalmTaintEscape,
            "psalm-taint-unescape" => TagKind::PsalmTaintUnescape,
            "psalmtaintunescape" => TagKind::PsalmTaintUnescape,
            "psalm-taint-specialize" => TagKind::PsalmTaintSpecialize,
            "psalmtaintspecialize" => TagKind::PsalmTaintSpecialize,
            "psalm-flow" => TagKind::PsalmFlow,
            "psalmflow" => TagKind::PsalmFlow,
            "psalm-type" => TagKind::PsalmType,
            "psalm-import-type" => TagKind::PsalmImportType,
            "psalm-require-extends" => TagKind::PsalmRequireExtends,
            "psalmrequireextends" => TagKind::PsalmRequireExtends,
            "psalm-require-implements" => TagKind::PsalmRequireImplements,
            "psalmrequireimplements" => TagKind::PsalmRequireImplements,
            "psalm-ignore-variable-property" => TagKind::PsalmIgnoreVariableProperty,
            "psalmignorevariableproperty" => TagKind::PsalmIgnoreVariableProperty,
            "psalm-ignore-variable-method" => TagKind::PsalmIgnoreVariableMethod,
            "psalmignorevariablemethod" => TagKind::PsalmIgnoreVariableMethod,
            "psalm-yield" => TagKind::PsalmYield,
            "phpstan-assert" => TagKind::PhpstanAssert,
            "phpstan-assert-if-true" => TagKind::PhpstanAssertIfTrue,
            "phpstan-assert-if-false" => TagKind::PhpstanAssertIfFalse,
            "phpstan-self-out" => TagKind::PhpstanSelfOut,
            "phpstan-this-out" => TagKind::PhpstanThisOut,
            "phpstan-require-extends" => TagKind::PhpstanRequireExtends,
            "phpstan-require-implements" => TagKind::PhpstanRequireImplements,
            "phpstan-template" => TagKind::PhpstanTemplate,
            "phpstan-template-covariant" => TagKind::PhpstanTemplateCovariant,
            "phpstan-template-contravariant" => TagKind::PhpstanTemplateContravariant,
            "phpstan-param" => TagKind::PhpstanParam,
            "phpstan-return" => TagKind::PhpstanReturn,
            "phpstan-var" => TagKind::PhpstanVar,
            "phpstan-readonly" => TagKind::PhpstanReadOnly,
            "phpstan-immutable" => TagKind::PhpstanImmutable,
            _ => TagKind::Other,
        }
    }
}
