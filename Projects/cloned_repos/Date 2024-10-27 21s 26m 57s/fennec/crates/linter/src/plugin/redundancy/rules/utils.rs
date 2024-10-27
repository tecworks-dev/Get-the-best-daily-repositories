use fennec_ast::*;

/// Determine if an expression is truthy.
///
/// This function will return true if the expression is truthy, and false otherwise.
///
/// When this function returns true, it does not necessarily mean that the expression will always evaluate to true.
/// It simply means that the expression is truthy in the context of PHP.
pub fn is_truthy(expression: &Expression) -> bool {
    match &expression {
        Expression::Parenthesized(parenthesized) => is_truthy(&parenthesized.expression),
        Expression::Referenced(referenced) => is_truthy(&referenced.expression),
        Expression::Suppressed(suppressed) => is_truthy(&suppressed.expression),
        Expression::Literal(literal) => match &literal {
            Literal::True(_) => true,
            _ => false,
        },
        Expression::CoalesceOperation(coalesce_operation) => is_truthy(&coalesce_operation.lhs),
        Expression::AnonymousClass(_) => true,
        Expression::Closure(_) => true,
        Expression::ArrowFunction(_) => true,
        Expression::Array(array) => !array.elements.is_empty(),
        Expression::LegacyArray(array) => !array.elements.is_empty(),
        Expression::ClosureCreation(_) => true,
        Expression::LogicalOperation(operation) => match operation.as_ref() {
            LogicalOperation::Prefix(logical_prefix_operation) => return is_falsy(&logical_prefix_operation.value),
            LogicalOperation::Infix(logical_infix_operation) => match &logical_infix_operation.operator {
                LogicalInfixOperator::LowPrecedenceAnd(_) | LogicalInfixOperator::And(_) => {
                    is_truthy(&logical_infix_operation.lhs) && is_truthy(&logical_infix_operation.rhs)
                }
                LogicalInfixOperator::LowPrecedenceOr(_) | LogicalInfixOperator::Or(_) => {
                    is_truthy(&logical_infix_operation.lhs) || is_truthy(&logical_infix_operation.rhs)
                }
                _ => false,
            },
        },
        Expression::AssignmentOperation(assignment) => is_truthy(&assignment.rhs),
        _ => false,
    }
}

/// Determine if an expression is falsy.
///
/// This function will return true if the expression is falsy, and false otherwise.
///
/// When this function returns false, it does not mean that the expression is truthy,
/// it just means that we could not determine if the expression is falsy.
pub fn is_falsy(expression: &Expression) -> bool {
    match &expression {
        Expression::Parenthesized(parenthesized) => is_falsy(&parenthesized.expression),
        Expression::Referenced(referenced) => is_falsy(&referenced.expression),
        Expression::Suppressed(suppressed) => is_falsy(&suppressed.expression),
        Expression::Literal(literal) => match &literal {
            Literal::False(_) | Literal::Null(_) => true,
            _ => false,
        },
        Expression::Array(array) => array.elements.is_empty(),
        Expression::LegacyArray(array) => array.elements.is_empty(),
        Expression::CoalesceOperation(coalesce_operation) => {
            is_falsy(&coalesce_operation.lhs) && is_falsy(&coalesce_operation.rhs)
        }
        Expression::LogicalOperation(operation) => match operation.as_ref() {
            LogicalOperation::Prefix(logical_prefix_operation) => return is_truthy(&logical_prefix_operation.value),
            LogicalOperation::Infix(logical_infix_operation) => match &logical_infix_operation.operator {
                LogicalInfixOperator::LowPrecedenceAnd(_) | LogicalInfixOperator::And(_) => {
                    is_falsy(&logical_infix_operation.lhs) || is_falsy(&logical_infix_operation.rhs)
                }
                LogicalInfixOperator::LowPrecedenceOr(_) | LogicalInfixOperator::Or(_) => {
                    is_falsy(&logical_infix_operation.lhs) && is_falsy(&logical_infix_operation.rhs)
                }
                _ => false,
            },
        },
        Expression::AssignmentOperation(assignment) => is_falsy(&assignment.rhs),
        _ => false,
    }
}

/// Determine if a statement contains only definitions.
pub fn statement_contains_only_definitions<'ast>(statement: &'ast Statement) -> bool {
    let (definitions, statements) = get_statement_stats(&statement);

    definitions != 0 && statements == 0
}

pub fn statement_sequence_contains_only_definitions<'ast>(statement: &'ast Sequence<Statement>) -> bool {
    let mut definitions = 0;
    let mut statements = 0;
    for statement in statement.iter() {
        let (def, stmt) = get_statement_stats(statement);

        definitions += def;
        statements += stmt;
    }

    definitions != 0 && statements == 0
}

fn get_statement_stats<'ast>(statement: &'ast Statement) -> (usize, usize) {
    let mut total_definitions = 0;
    let mut total_statements = 0;

    match &statement {
        Statement::Namespace(namespace) => {
            for statement in namespace.statements().iter() {
                let (definitions, statements) = get_statement_stats(statement);
                total_definitions += definitions;
                total_statements += statements;
            }
        }
        Statement::Block(block) => {
            for statement in block.statements.iter() {
                let (definitions, statements) = get_statement_stats(statement);
                total_definitions += definitions;
                total_statements += statements;
            }
        }
        Statement::Class(_)
        | Statement::Interface(_)
        | Statement::Trait(_)
        | Statement::Enum(_)
        | Statement::Function(_)
        | Statement::Constant(_) => {
            total_definitions += 1;
        }
        _ => {
            total_statements += 1;
        }
    }

    (total_definitions, total_statements)
}
