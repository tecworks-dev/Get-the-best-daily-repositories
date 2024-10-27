use fennec_ast::ast::*;
use fennec_ast::Node;
use fennec_ast::Program;
use fennec_interner::StringIdentifier;
use fennec_interner::ThreadedInterner;
use fennec_names::Names;
use fennec_reporting::Issue;
use fennec_reporting::IssueCollection;
use fennec_span::HasSpan;
use fennec_span::Position;
use fennec_span::Span;

#[derive(Debug)]
pub struct Context<'a> {
    interner: &'a ThreadedInterner,
    program: &'a Program,
    names: &'a Names,
    issues: IssueCollection,
    ancestors: Vec<Span>,
}

impl<'a> Context<'a> {
    pub fn new(interner: &'a ThreadedInterner, program: &'a Program, names: &'a Names) -> Self {
        Self { interner, program, names, issues: IssueCollection::default(), ancestors: vec![] }
    }

    pub fn program(&self) -> Node<'a> {
        Node::Program(self.program)
    }

    pub fn report(&mut self, issue: Issue) {
        self.issues.push(issue);
    }

    pub fn intern(&self, string: impl AsRef<str>) -> StringIdentifier {
        self.interner.intern(string)
    }

    pub fn lookup(&self, id: StringIdentifier) -> String {
        self.interner.lookup(id).to_string()
    }

    pub fn lookup_name(&self, position: &Position) -> String {
        self.lookup(self.names.get(position))
    }

    pub fn lookup_hint(&self, hint: &Hint) -> String {
        match hint {
            Hint::Identifier(identifier) => self.lookup_name(&identifier.span().start),
            Hint::Parenthesized(parenthesized_hint) => {
                format!("({})", self.lookup_hint(&parenthesized_hint.hint))
            }
            Hint::Nullable(nullable_hint) => format!("?{}", self.lookup_hint(&nullable_hint.hint)),
            Hint::Union(union_hint) => {
                format!("{}|{}", self.lookup_hint(&union_hint.left), self.lookup_hint(&union_hint.right))
            }
            Hint::Intersection(intersection_hint) => {
                format!("{}&{}", self.lookup_hint(&intersection_hint.left), self.lookup_hint(&intersection_hint.right))
            }
            Hint::Null(keyword) => self.lookup(keyword.value),
            Hint::True(keyword) => self.lookup(keyword.value),
            Hint::False(keyword) => self.lookup(keyword.value),
            Hint::Array(keyword) => self.lookup(keyword.value),
            Hint::Callable(keyword) => self.lookup(keyword.value),
            Hint::Static(keyword) => self.lookup(keyword.value),
            Hint::Self_(keyword) => self.lookup(keyword.value),
            Hint::Parent(keyword) => self.lookup(keyword.value),
            Hint::Void(identifier) => self.lookup(identifier.value),
            Hint::Never(identifier) => self.lookup(identifier.value),
            Hint::Float(identifier) => self.lookup(identifier.value),
            Hint::Bool(identifier) => self.lookup(identifier.value),
            Hint::Integer(identifier) => self.lookup(identifier.value),
            Hint::String(identifier) => self.lookup(identifier.value),
            Hint::Object(identifier) => self.lookup(identifier.value),
            Hint::Mixed(identifier) => self.lookup(identifier.value),
            Hint::Iterable(identifier) => self.lookup(identifier.value),
        }
    }

    pub fn push_ancestor(&mut self, node: Span) {
        self.ancestors.push(node);
    }

    pub fn get_ancestors_len(&self) -> usize {
        self.ancestors.len()
    }

    pub fn get_ancestor(&self, index: usize) -> Span {
        self.ancestors[index]
    }

    pub fn pop_ancestor(&mut self) {
        self.ancestors.pop();
    }

    pub fn take_issue_collection(self) -> IssueCollection {
        self.issues
    }
}
