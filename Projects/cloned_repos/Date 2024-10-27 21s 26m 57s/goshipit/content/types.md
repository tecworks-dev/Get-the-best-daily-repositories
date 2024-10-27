# Types

## Introduction

This is a collection of types (structs) that the components use as input arguments. We're using structs as they provide a convenient way to pass default values to the components; struct fields initialize to the respective type's default value. We can use this feature to make some of the fields of a struct optional. For example, we can pass the `model.Anchor` argument to an anchor element, `<a>`:

```go
templ Anchor(anchor model.Anchor) {
    <a
        if anchor.Href != "" {
            href={ anchor.Href }
        }
        { anchor.Attrs... }
    >
        if anchor.Icon != nil {
            @anchor.Icon
        }
        { anchor.Label }
    </a>
}
```

Here we can define the anchor element to optionally have an `href` attribute, or if we choose, a `hx-get` instead:

```go
@Anchor(model.Anchor{Href: "/"})
@Anchor(model.Anchor{Attrs: templ.Attributes{"hx-get": "/"}})
```

## The types

You can place these types in a file e.g. `internal/model/components.go` for easily accessing the correct type for each component.
