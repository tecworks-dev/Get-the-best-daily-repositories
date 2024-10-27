use fennec_ast::*;
use fennec_interner::StringIdentifier;
use fennec_walker::MutWalker;

use crate::context::LintContext;

/// Determine if a variable is a super global variable.
pub fn is_super_global_variable(name: &str) -> bool {
    matches!(
        name,
        "$_GET" | "$_POST" | "$_COOKIE" | "$_REQUEST" | "$_SERVER" | "$_FILES" | "$_ENV" | "$_SESSION" | "$GLOBALS"
    )
}

/// Determine if a variable is a predefined variable.
///
/// Predefined variables are variables that are defined by PHP.
pub fn is_predefined_variable(name: &str) -> bool {
    is_super_global_variable(name) || "$this" == name
}

/// Determine if the expression uses the given variable.
pub fn is_variable_used_in_expression<'ast>(
    expression: &'ast Expression,
    context: &LintContext<'_>,
    variable: StringIdentifier,
) -> bool {
    use crate::plugin::fennec::rules::utils::internal::VariableReference;
    use crate::plugin::fennec::rules::utils::internal::VariableWalker;

    let mut context = (Vec::default(), context);
    let mut walker = VariableWalker { in_closure: false };

    walker.walk_expression(expression, &mut context);

    let variables = context.0;
    for variable_reference in variables {
        match variable_reference {
            VariableReference::Use(string_identifier) => {
                if string_identifier == variable {
                    return true;
                }
            }
            VariableReference::Assign(string_identifier) => {
                // the variable was re-assigned before it was used
                if string_identifier == variable {
                    return false;
                }
            }
            VariableReference::Unset(string_identifier) => {
                // the variable was unset before it was used
                if string_identifier == variable {
                    return false;
                }
            }
        }
    }

    false
}

/// Given a block, get all the variable names that are used in the block before they are declared.
pub fn get_foreign_variable_names<'ast>(block: &'ast Block, context: &LintContext<'_>) -> Vec<StringIdentifier> {
    use crate::plugin::fennec::rules::utils::internal::VariableReference;
    use crate::plugin::fennec::rules::utils::internal::VariableWalker;

    // we must use a vec here instead of a set because we need to preserve the order of the variables
    // in order to determine the order in which they are used/delcared.
    let mut context = (Vec::default(), context);
    let mut walker = VariableWalker { in_closure: false };

    walker.walk_block(block, &mut context);

    let variables = context.0;
    let mut assigned = Vec::new();
    let mut foreign = Vec::new();
    for variable in variables {
        match &variable {
            VariableReference::Use(string_identifier) => {
                if !assigned.contains(string_identifier) && !foreign.contains(string_identifier) {
                    // the variable is used before it is assigned
                    foreign.push(*string_identifier);
                }
            }
            VariableReference::Assign(string_identifier) => {
                assigned.push(*string_identifier);
            }
            VariableReference::Unset(string_identifier) => {
                assigned.retain(|assigned| assigned != string_identifier);
            }
        }
    }

    foreign
}

/// A utility function to get the content of a comment trivia.
///
/// This function will return the content of a comment trivia, without the comment markers.
pub fn comment_content<'ast>(trivia: &'ast Trivia, context: &LintContext<'_>) -> Option<String> {
    match trivia.kind {
        TriviaKind::MultiLineComment => {
            let content = context.lookup(trivia.value);
            let content = &content[2..content.len() - 2];

            Some(remove_star_prefix(content))
        }
        TriviaKind::DocBlockComment => {
            let content = context.lookup(trivia.value);
            let content = &content[3..content.len() - 2];

            Some(remove_star_prefix(content))
        }
        TriviaKind::SingleLineComment => {
            let content = context.lookup(trivia.value);

            Some(content[2..].to_string())
        }
        TriviaKind::HashComment => {
            let content = context.lookup(trivia.value);

            Some(content[1..].to_string())
        }
        TriviaKind::WhiteSpace => None,
    }
}

fn remove_star_prefix(content: &str) -> String {
    let mut lines = content.lines().map(remove_stared_line_prefix);

    let mut result = String::new();
    if let Some(first) = lines.next() {
        result.push_str(first);
    }

    for line in lines {
        result.push_str("\n");
        result.push_str(line);
    }

    result
}

fn remove_stared_line_prefix(line: &str) -> &str {
    let trimmed = line.trim_start();

    if trimmed.starts_with('*') {
        trimmed[1..].trim_start()
    } else {
        line
    }
}

mod internal {
    use super::is_predefined_variable;

    use fennec_ast::*;
    use fennec_interner::StringIdentifier;
    use fennec_walker::MutWalker;

    use crate::context::LintContext;

    #[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, PartialOrd, Ord)]
    pub(super) enum VariableReference {
        Use(StringIdentifier),
        Assign(StringIdentifier),
        Unset(StringIdentifier),
    }

    #[derive(Debug)]
    pub(super) struct VariableWalker {
        pub in_closure: bool,
    }

    impl<'a> MutWalker<(Vec<VariableReference>, &'a LintContext<'a>)> for VariableWalker {
        fn walk_in_foreach_value_target<'ast>(
            &mut self,
            foreach_value_target: &'ast ForeachValueTarget,
            context: &mut (Vec<VariableReference>, &'a LintContext<'a>),
        ) {
            if self.in_closure {
                return;
            }

            context.0.extend(scan_expression_for_assignment(&foreach_value_target.value, &context.1));
        }

        fn walk_in_foreach_key_value_target<'ast>(
            &mut self,
            foreach_key_value_target: &'ast ForeachKeyValueTarget,
            context: &mut (Vec<VariableReference>, &'a LintContext<'a>),
        ) {
            if self.in_closure {
                return;
            }

            context.0.extend(scan_expression_for_assignment(&foreach_key_value_target.key, &context.1));

            context.0.extend(scan_expression_for_assignment(&foreach_key_value_target.value, &context.1));
        }

        fn walk_in_try_catch_clause<'ast>(
            &mut self,
            try_catch_clause: &'ast TryCatchClause,
            context: &mut (Vec<VariableReference>, &'a LintContext<'a>),
        ) {
            if self.in_closure {
                return;
            }

            if let Some(variable) = &try_catch_clause.variable {
                context.0.push(VariableReference::Assign(variable.name));
            }
        }

        fn walk_in_static_concrete_item<'ast>(
            &mut self,
            static_concrete_item: &'ast StaticConcreteItem,
            context: &mut (Vec<VariableReference>, &'a LintContext<'a>),
        ) {
            if self.in_closure {
                return;
            }

            context.0.push(VariableReference::Assign(static_concrete_item.variable.name));
        }

        fn walk_in_static_abstract_item<'ast>(
            &mut self,
            static_abstract_item: &'ast StaticAbstractItem,
            context: &mut (Vec<VariableReference>, &'a LintContext<'a>),
        ) {
            if self.in_closure {
                return;
            }

            context.0.push(VariableReference::Assign(static_abstract_item.variable.name));
        }

        fn walk_in_global<'ast>(
            &mut self,
            global: &'ast Global,
            context: &mut (Vec<VariableReference>, &'a LintContext<'a>),
        ) {
            if self.in_closure {
                return;
            }

            for variable in global.variables.iter() {
                let Variable::Direct(variable) = variable else {
                    continue;
                };

                context.0.push(VariableReference::Assign(variable.name));
            }
        }

        fn walk_assignment_operation<'ast>(
            &mut self,
            assignment_operation: &'ast AssignmentOperation,
            context: &mut (Vec<VariableReference>, &'a LintContext<'a>),
        ) {
            // we need to walk the right hand side first to ensure that we don't
            // mark variables as being assigned to when they are only being used
            // in the right hand side.
            self.walk_expression(&assignment_operation.rhs, context);

            if self.in_closure {
                return;
            }

            let variables = scan_expression_for_assignment(&assignment_operation.lhs, &context.1);

            match assignment_operation.operator {
                AssignmentOperator::Assign(_) => {
                    context.0.extend(variables);
                }
                _ => {
                    for variable in variables {
                        if let VariableReference::Assign(name) = variable {
                            context.0.push(VariableReference::Use(name));
                            context.0.push(VariableReference::Assign(name));
                        } else {
                            context.0.push(variable);
                        }
                    }
                }
            }

            // then we walk the left hand side
            self.walk_expression(&assignment_operation.lhs, context);
        }

        fn walk_in_direct_variable<'ast>(
            &mut self,
            direct_variable: &'ast DirectVariable,
            context: &mut (Vec<VariableReference>, &'a LintContext<'a>),
        ) {
            if self.in_closure {
                return;
            }

            let name = context.1.interner.lookup(direct_variable.name);
            if !is_predefined_variable(name) {
                context.0.push(VariableReference::Use(direct_variable.name));
            }
        }

        fn walk_in_closure<'ast>(
            &mut self,
            closure: &'ast Closure,
            context: &mut (Vec<VariableReference>, &'a LintContext<'a>),
        ) {
            if let Some(use_clause) = &closure.use_clause {
                for use_clause_variable in use_clause.variables.iter() {
                    context.0.push(VariableReference::Use(use_clause_variable.variable.name));
                }
            }

            self.in_closure = true;
        }

        fn walk_out_closure<'ast>(
            &mut self,
            _closure: &'ast Closure,
            _context: &mut (Vec<VariableReference>, &'a LintContext<'a>),
        ) {
            self.in_closure = false;
        }

        fn walk_in_arrow_function<'ast>(
            &mut self,
            arrow_function: &'ast ArrowFunction,
            context: &mut (Vec<VariableReference>, &'a LintContext<'a>),
        ) {
            for parameter in arrow_function.parameters.parameters.iter() {
                context.0.push(VariableReference::Assign(parameter.variable.name));
            }
        }

        fn walk_out_arrow_function<'ast>(
            &mut self,
            arrow_function: &'ast ArrowFunction,
            context: &mut (Vec<VariableReference>, &'a LintContext<'a>),
        ) {
            for parameter in arrow_function.parameters.parameters.iter() {
                context.0.push(VariableReference::Unset(parameter.variable.name));
            }
        }
    }

    fn scan_expression_for_assignment<'ast>(
        expression: &'ast Expression,
        context: &LintContext<'_>,
    ) -> Vec<VariableReference> {
        let mut variables = Vec::default();

        match &expression {
            Expression::Variable(variable) => {
                let Variable::Direct(variable) = variable else {
                    return variables;
                };

                let name = context.interner.lookup(variable.name);
                if !is_predefined_variable(name) {
                    variables.push(VariableReference::Assign(variable.name));
                }
            }
            Expression::Array(array) => {
                for element in array.elements.iter() {
                    match &element {
                        ArrayElement::KeyValue(key_value_array_element) => {
                            variables.extend(scan_expression_for_assignment(&key_value_array_element.key, context));
                            variables.extend(scan_expression_for_assignment(&key_value_array_element.value, context));
                        }
                        ArrayElement::Value(value_array_element) => {
                            variables.extend(scan_expression_for_assignment(&value_array_element.value, context));
                        }
                        _ => {}
                    }
                }
            }
            Expression::LegacyArray(array) => {
                for element in array.elements.iter() {
                    match &element {
                        ArrayElement::KeyValue(key_value_array_element) => {
                            variables.extend(scan_expression_for_assignment(&key_value_array_element.key, context));
                            variables.extend(scan_expression_for_assignment(&key_value_array_element.value, context));
                        }
                        ArrayElement::Value(value_array_element) => {
                            variables.extend(scan_expression_for_assignment(&value_array_element.value, context));
                        }
                        _ => {}
                    }
                }
            }
            Expression::List(list) => {
                for element in list.elements.iter() {
                    match &element {
                        ArrayElement::KeyValue(key_value_array_element) => {
                            variables.extend(scan_expression_for_assignment(&key_value_array_element.key, context));
                            variables.extend(scan_expression_for_assignment(&key_value_array_element.value, context));
                        }
                        ArrayElement::Value(value_array_element) => {
                            variables.extend(scan_expression_for_assignment(&value_array_element.value, context));
                        }
                        _ => {}
                    }
                }
            }
            Expression::ArrayAppend(append) => {
                // PHP treats `$a[] = 1;` the same as `$a = []; $a[] = 1;` if `$a` is not defined.
                variables.extend(scan_expression_for_assignment(&append.array, context));
            }
            Expression::ArrayAccess(array_access) => {
                // PHP treats `$a[$b] = 1;` the same as `$a = []; $a[$b] = 1;` if `$a` is not defined.
                variables.extend(scan_expression_for_assignment(&array_access.array, context));
            }
            _ => {}
        }

        variables
    }
}
