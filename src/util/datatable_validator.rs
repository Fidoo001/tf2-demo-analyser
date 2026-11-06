use tf_demo_parser::demo::packet::datatable::{ParseSendTable, ServerClass};
use tf_demo_parser::demo::sendprop::SendPropFlag;
use tf_demo_parser::ParseError;

/// Source engine specifications and practical limits:
/// - Official Source engine MAX_DATATABLE_PROPS is 4096.
/// - In legitimate TF2 demos, DT_TFPlayerResource reaches ~3,675 props.
/// - Legitimate demos have ~500 tables and ~350 server classes.
/// - Legitimate demos have ~62,000 total flattened properties across all tables.
pub const MAX_DATATABLES: usize = 2048;
pub const MAX_SERVER_CLASSES: usize = 2048;
pub const MAX_DATATABLE_PROPS: usize = 4096;
pub const MAX_DATATABLE_DEPTH: usize = 32;
pub const MAX_TOTAL_DATATABLE_PROPS: usize = 250_000;

/// Validates that a demo's SendTables and ServerClasses conform to Source engine
/// limits and do not contain exponential expansion DAG bombs (demo zip bombs).
pub fn validate_send_tables(
    tables: &[ParseSendTable],
    server_classes: &[ServerClass],
) -> Result<(), ParseError> {
    if tables.len() > MAX_DATATABLES {
        return Err(ParseError::InvalidDemo(
            "Too many datatables in demo (exceeds MAX_DATATABLES limit)",
        ));
    }
    if server_classes.len() > MAX_SERVER_CLASSES {
        return Err(ParseError::InvalidDemo(
            "Too many server classes in demo (exceeds MAX_SERVER_CLASSES limit)",
        ));
    }

    let mut total_props: usize = 0;

    for table in tables {
        let mut count: usize = 0;
        let mut stack: Vec<&str> = Vec::with_capacity(MAX_DATATABLE_DEPTH);
        count_table_props(table, tables, &mut stack, &mut count)?;

        total_props = total_props.saturating_add(count);
        if total_props > MAX_TOTAL_DATATABLE_PROPS {
            return Err(ParseError::InvalidDemo(
                "Total datatable properties exceed safe limits (potential zip bomb attack)",
            ));
        }
    }

    Ok(())
}

/// Recursively computes an upper bound on flattened properties without allocating memory.
/// Immediately aborts if property count exceeds MAX_DATATABLE_PROPS or if depth exceeds MAX_DATATABLE_DEPTH.
fn count_table_props<'a>(
    table: &'a ParseSendTable,
    tables: &'a [ParseSendTable],
    stack: &mut Vec<&'a str>,
    count: &mut usize,
) -> Result<(), ParseError> {
    if stack.len() >= MAX_DATATABLE_DEPTH {
        return Err(ParseError::InvalidDemo(
            "DataTable nesting depth exceeded safe limit (recursion or cycle detected)",
        ));
    }

    let table_name = table.name.as_str();
    if stack.contains(&table_name) {
        return Err(ParseError::InvalidDemo(
            "Cyclic DataTable reference detected in demo",
        ));
    }

    stack.push(table_name);

    for prop in &table.props {
        if prop.flags.contains(SendPropFlag::Exclude) {
            continue;
        }

        if let Some(target) = prop.get_data_table(tables) {
            count_table_props(target, tables, stack, count)?;
        } else {
            *count += 1;
            if *count > MAX_DATATABLE_PROPS {
                return Err(ParseError::InvalidDemo(
                    "DataTable property expansion limit exceeded (potential demo zip bomb)",
                ));
            }
        }
    }

    stack.pop();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tf_demo_parser::demo::packet::datatable::SendTableName;
    use tf_demo_parser::demo::sendprop::{
        RawSendPropDefinition, SendPropFlags, SendPropIdentifier, SendPropType,
    };

    fn create_leaf_prop(owner: &'static str, name: &'static str) -> RawSendPropDefinition {
        RawSendPropDefinition {
            prop_type: SendPropType::Int,
            name: name.into(),
            identifier: SendPropIdentifier::new(owner, name),
            flags: SendPropFlags::default(),
            table_name: None,
            low_value: None,
            high_value: None,
            bit_count: Some(32),
            element_count: None,
            array_property: None,
            original_bit_count: Some(32),
        }
    }

    fn create_dt_prop(
        owner: &'static str,
        prop_name: &'static str,
        target_table: &'static str,
    ) -> RawSendPropDefinition {
        RawSendPropDefinition {
            prop_type: SendPropType::DataTable,
            name: prop_name.into(),
            identifier: SendPropIdentifier::new(owner, prop_name),
            flags: SendPropFlags::default(),
            table_name: Some(SendTableName::from(target_table)),
            low_value: None,
            high_value: None,
            bit_count: None,
            element_count: None,
            array_property: None,
            original_bit_count: None,
        }
    }

    #[test]
    fn test_valid_simple_tables() {
        let t1 = ParseSendTable {
            name: "t1".into(),
            props: vec![create_leaf_prop("t1", "p1"), create_leaf_prop("t1", "p2")],
            needs_decoder: true,
        };
        let t0 = ParseSendTable {
            name: "t0".into(),
            props: vec![create_dt_prop("t0", "sub", "t1")],
            needs_decoder: true,
        };
        let tables = vec![t0, t1];
        assert!(validate_send_tables(&tables, &[]).is_ok());
    }

    #[test]
    fn test_cycle_detected() {
        let t0 = ParseSendTable {
            name: "t0".into(),
            props: vec![create_dt_prop("t0", "sub1", "t1")],
            needs_decoder: true,
        };
        let t1 = ParseSendTable {
            name: "t1".into(),
            props: vec![create_dt_prop("t1", "sub0", "t0")],
            needs_decoder: true,
        };
        let tables = vec![t0, t1];
        let result = validate_send_tables(&tables, &[]);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("Cyclic") || err_msg.contains("depth"));
    }

    #[test]
    fn test_property_explosion_blocked() {
        // Table t1 with 100 leaves
        let mut t1_props = Vec::new();
        for i in 0..100 {
            t1_props.push(create_leaf_prop("t1", Box::leak(format!("leaf_{}", i).into_boxed_str())));
        }
        let t1 = ParseSendTable {
            name: "t1".into(),
            props: t1_props,
            needs_decoder: true,
        };

        // Table t0 with 50 refs to t1 -> 50 * 100 = 5000 props (> 4096)
        let mut t0_props = Vec::new();
        for i in 0..50 {
            t0_props.push(create_dt_prop("t0", Box::leak(format!("dt_{}", i).into_boxed_str()), "t1"));
        }
        let t0 = ParseSendTable {
            name: "t0".into(),
            props: t0_props,
            needs_decoder: true,
        };

        let tables = vec![t0, t1];
        let result = validate_send_tables(&tables, &[]);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("expansion limit exceeded"));
    }

    #[test]
    fn test_too_many_tables_blocked() {
        let mut tables = Vec::new();
        for i in 0..(MAX_DATATABLES + 1) {
            tables.push(ParseSendTable {
                name: SendTableName::from(format!("t_{}", i)),
                props: vec![],
                needs_decoder: false,
            });
        }
        assert!(validate_send_tables(&tables, &[]).is_err());
    }
}
