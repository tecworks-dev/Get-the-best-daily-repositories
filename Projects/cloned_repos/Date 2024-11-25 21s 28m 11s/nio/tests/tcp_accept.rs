#![warn(rust_2018_idioms)]
#![cfg(all(feature = "full", not(target_os = "wasi")))] // Wasi doesn't support bind
#![cfg(not(miri))]

use futures::FutureExt;
use tokio::sync::{mpsc, oneshot};

use nio::net::{TcpListener, TcpStream};
use tokio_test::assert_ok;

use std::io;
use std::net::{IpAddr, SocketAddr};

macro_rules! test_accept {
    ($(($ident:ident, $target:expr),)*) => {
        $(
            #[nio::test]
            async fn $ident() {
                let listener = assert_ok!(TcpListener::bind($target).await);
                let addr = listener.local_addr().unwrap();

                let (tx, rx) = oneshot::channel();

                nio::spawn(async move {
                    let (socket, _) = assert_ok!(listener.accept().await);
                    assert_ok!(tx.send(socket));
                });

                let cli = assert_ok!(TcpStream::connect(&addr).await);
                let srv = assert_ok!(rx.await);

                assert_eq!(cli.local_addr().unwrap(), srv.peer_addr().unwrap());
            }
        )*
    }
}

test_accept! {
    (ip_str, "127.0.0.1:0"),
    (host_str, "localhost:0"),
    (socket_addr, "127.0.0.1:0".parse::<SocketAddr>().unwrap()),
    (str_port_tuple, ("127.0.0.1", 0)),
    (ip_port_tuple, ("127.0.0.1".parse::<IpAddr>().unwrap(), 0)),
}

use std::pin::Pin;
use std::sync::{
    atomic::{AtomicUsize, Ordering::SeqCst},
    Arc,
};
use std::task::{Context, Poll};
use tokio_stream::{Stream, StreamExt};

struct TrackPolls<'a> {
    npolls: Arc<AtomicUsize>,
    listener: &'a mut TcpListener,
}

impl<'a> Stream for TrackPolls<'a> {
    type Item = io::Result<(TcpStream, SocketAddr)>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.npolls.fetch_add(1, SeqCst);
        self.listener.accept().poll_unpin(cx).map(Some)
    }
}

#[nio::test]
async fn no_extra_poll() {
    let mut listener = assert_ok!(TcpListener::bind("127.0.0.1:0").await);
    let addr = listener.local_addr().unwrap();

    let (tx, rx) = oneshot::channel();
    let (accepted_tx, mut accepted_rx) = mpsc::unbounded_channel();

    nio::spawn(async move {
        let mut incoming = TrackPolls {
            npolls: Arc::new(AtomicUsize::new(0)),
            listener: &mut listener,
        };
        assert_ok!(tx.send(Arc::clone(&incoming.npolls)));
        while incoming.next().await.is_some() {
            accepted_tx.send(()).unwrap();
        }
    });

    let npolls = assert_ok!(rx.await);
    nio::task::yield_now().await;

    // should have been polled exactly once: the initial poll
    assert_eq!(npolls.load(SeqCst), 1);

    let _ = assert_ok!(TcpStream::connect(&addr).await);
    accepted_rx.recv().await.unwrap();

    // should have been polled twice more: once to yield Some(), then once to yield Pending
    assert_eq!(npolls.load(SeqCst), 1 + 2);
}
