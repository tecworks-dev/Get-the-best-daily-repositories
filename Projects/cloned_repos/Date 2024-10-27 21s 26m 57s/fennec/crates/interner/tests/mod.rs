use fennec_interner::Interner;
use fennec_interner::StringIdentifier;
use fennec_interner::ThreadedInterner;

#[test]
fn test_empty_is_always_zero() {
    assert_eq!(StringIdentifier::empty().value(), 0);

    let mut interner = Interner::new();

    assert_eq!(interner.intern(""), StringIdentifier::empty());

    let threaded_interner = ThreadedInterner::new();

    assert_eq!(threaded_interner.intern(""), StringIdentifier::empty());
}

#[test]
fn test_interner() {
    let interner = ThreadedInterner::new();

    assert_eq!(interner.len(), 0);
    assert!(interner.is_empty());

    // Test basic interning and lookup
    let id = interner.intern("hello");
    assert_eq!(interner.lookup(id), "hello");

    let id = interner.intern("hello");
    assert_eq!(interner.lookup(id), "hello");

    let id = interner.intern("world");
    assert_eq!(interner.lookup(id), "world");

    let id = interner.intern("hello");
    assert_eq!(interner.lookup(id), "hello");

    assert_eq!(interner.len(), 2);
    assert!(!interner.is_empty());
}

#[test]
fn test_interner_multithreaded() {
    let interner = ThreadedInterner::new();

    let mut handles = Vec::new();
    for i in 0..20 {
        handles.push(std::thread::spawn({
            let interner = interner.clone();

            move || {
                (
                    interner.intern("apple"),
                    interner.intern("banana"),
                    interner.intern("cherry"),
                    interner.intern("coconut"),
                    interner.intern("durian"),
                    interner.intern(format!("hello-{}", i)),
                )
            }
        }));
    }

    let mut sets = Vec::new();
    for (index, handle) in handles.into_iter().enumerate() {
        let set = handle.join().unwrap();

        assert_eq!(interner.lookup(set.0), "apple");
        assert_eq!(interner.lookup(set.1), "banana");
        assert_eq!(interner.lookup(set.2), "cherry");
        assert_eq!(interner.lookup(set.3), "coconut");
        assert_eq!(interner.lookup(set.4), "durian");
        assert_eq!(interner.lookup(set.5), format!("hello-{}", index));

        sets.push((set.0, set.1, set.2, set.3, set.4));
    }

    assert_eq!(interner.len(), 25);

    let first_set = sets.first().unwrap();
    for set in sets.iter() {
        assert_eq!(first_set, set);
    }
}
