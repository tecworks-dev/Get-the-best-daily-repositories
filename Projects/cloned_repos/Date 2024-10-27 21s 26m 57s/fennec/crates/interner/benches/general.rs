use std::sync::Arc;
use std::thread;

use criterion::black_box;
use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;

use fennec_interner::Interner;
use fennec_interner::ThreadedInterner;

fn bench_current_thread_intern(c: &mut Criterion) {
    let mut group = c.benchmark_group("Interner::intern");
    let mut interner = Interner::new();

    // Test with varying sizes of input data
    for size in [100, 1_000, 10_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let strings: Vec<String> = (0..size).map(|i| format!("string-{}", i)).collect();

            b.iter(|| {
                for s in &strings {
                    black_box(interner.intern(s));
                }
            });
        });
    }

    group.finish();
}

fn bench_current_thread_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("Interner::get");
    let mut interner = Interner::new();
    let size = 10_000;

    // Pre-intern some strings
    let strings: Vec<String> = (0..size).map(|i| format!("string-{}", i)).collect();
    for s in &strings {
        interner.intern(s);
    }

    // Test get method with existing and non-existing strings
    group.throughput(Throughput::Elements(size as u64 * 2));
    group.bench_function("get_existing_and_new_strings", |b| {
        let mut i = 0;
        b.iter(|| {
            // Existing string
            black_box(interner.get(&strings[i % size]));
            // Non-existing string
            black_box(interner.get(&format!("nonexistent-{}", i)));
            i += 1;
        });
    });

    group.finish();
}

fn bench_current_thread_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("Interner::lookup");
    let mut interner = Interner::new();
    let size = 10_000;

    // Intern strings and collect their identifiers
    let strings: Vec<String> = (0..size).map(|i| format!("string-{}", i)).collect();
    let identifiers: Vec<_> = strings.iter().map(|s| interner.intern(s)).collect();

    // Test lookup method with valid and invalid identifiers
    group.throughput(Throughput::Elements(size as u64 * 2));
    group.bench_function("lookup_valid_and_invalid_ids", |b| {
        let mut i = 0;
        b.iter(|| {
            // Valid identifier
            black_box(interner.lookup(identifiers[i % size]));
            // Invalid identifier (assuming higher than any assigned ID)
            black_box(interner.lookup(fennec_interner::StringIdentifier::new(usize::MAX - i)));
            i += 1;
        });
    });

    group.finish();
}

fn bench_threaded_interner_single_thread_intern(c: &mut Criterion) {
    let mut group = c.benchmark_group("ThreadedInterner::intern (Single Thread)");
    let interner = ThreadedInterner::new();

    // Test with varying sizes of input data
    for size in [100, 1_000, 10_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("SingleThread", size), size, |b, &size| {
            let strings: Vec<String> = (0..size).map(|i| format!("string-{}", i)).collect();

            b.iter(|| {
                for s in &strings {
                    black_box(interner.intern(s));
                }
            });
        });
    }

    group.finish();
}

fn bench_threaded_interner_multi_thread_intern(c: &mut Criterion) {
    let mut group = c.benchmark_group("ThreadedInterner::intern (Multi Thread)");
    let interner = Arc::new(ThreadedInterner::new());

    // Test with varying sizes of input data and thread counts
    for &(size, thread_count) in &[(1000, 4), (10000, 8), (100000, 16)] {
        group.throughput(Throughput::Elements((size * thread_count) as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("Threads: {}", thread_count), size),
            &(size, thread_count),
            |b, &(size, thread_count)| {
                let strings: Vec<String> = (0..size).map(|i| format!("string-{}", i)).collect();

                b.iter(|| {
                    let mut handles = Vec::with_capacity(thread_count);
                    for thread_id in 0..thread_count {
                        let interner = Arc::clone(&interner);
                        let strings = strings.clone();
                        let handle = thread::spawn(move || {
                            for s in strings.iter().skip(thread_id).step_by(thread_count) {
                                black_box(interner.intern(s));
                            }
                        });
                        handles.push(handle);
                    }

                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_threaded_interner_single_thread_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("ThreadedInterner::lookup (Single Thread)");
    let interner = ThreadedInterner::new();
    let size = 10_000;

    // Pre-intern strings
    let strings: Vec<String> = (0..size).map(|i| format!("string-{}", i)).collect();
    let identifiers: Vec<_> = strings.iter().map(|s| interner.intern(s)).collect();

    group.throughput(Throughput::Elements(size as u64));
    group.bench_function("lookup_valid_ids", |b| {
        let mut i = 0;
        b.iter(|| {
            black_box(interner.lookup(identifiers[i % size]));
            i += 1;
        });
    });

    group.finish();
}

fn bench_threaded_interner_multi_thread_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("ThreadedInterner::lookup (Multi Thread)");
    let interner = Arc::new(ThreadedInterner::new());
    let size = 100_000;
    let thread_count = 8;

    // Pre-intern strings
    let strings: Vec<String> = (0..size).map(|i| format!("string-{}", i)).collect();
    let identifiers: Vec<_> = strings.iter().map(|s| interner.intern(s)).collect();

    group.throughput(Throughput::Elements((size * thread_count) as u64));
    group.bench_function(&format!("Threads: {}", thread_count), |b| {
        b.iter(|| {
            let mut handles = Vec::with_capacity(thread_count);
            for thread_id in 0..thread_count {
                let interner = Arc::clone(&interner);
                let identifiers = identifiers.clone();
                let handle = thread::spawn(move || {
                    let mut i = thread_id;
                    while i < identifiers.len() {
                        black_box(interner.lookup(identifiers[i]));
                        i += thread_count;
                    }
                });
                handles.push(handle);
            }

            for handle in handles {
                handle.join().unwrap();
            }
        });
    });

    group.finish();
}

criterion_group!(current_thread, bench_current_thread_intern, bench_current_thread_get, bench_current_thread_lookup,);

criterion_group!(
    threaded_interner,
    bench_threaded_interner_single_thread_intern,
    bench_threaded_interner_multi_thread_intern,
    bench_threaded_interner_single_thread_lookup,
    bench_threaded_interner_multi_thread_lookup,
);

criterion_main!(current_thread, threaded_interner);
