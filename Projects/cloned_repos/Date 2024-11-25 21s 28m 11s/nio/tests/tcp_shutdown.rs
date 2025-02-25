#![warn(rust_2018_idioms)]
#![cfg(all(feature = "full", not(target_os = "wasi"), not(miri)))] // Wasi doesn't support bind

use nio::net::{TcpListener, TcpStream};
use tokio::io::{self, AsyncReadExt, AsyncWriteExt};
use tokio_test::assert_ok;

#[nio::test]
async fn shutdown() {
    let srv = assert_ok!(TcpListener::bind("127.0.0.1:0").await);
    let addr = assert_ok!(srv.local_addr());

    nio::spawn(async move {
        let mut stream = assert_ok!(TcpStream::connect(&addr).await);

        assert_ok!(AsyncWriteExt::shutdown(&mut stream).await);

        let mut buf = [0u8; 1];
        let n = assert_ok!(stream.read(&mut buf).await);
        assert_eq!(n, 0);
    });

    let (mut stream, _) = assert_ok!(srv.accept().await);
    let (mut rd, mut wr) = stream.split();

    let n = assert_ok!(io::copy(&mut rd, &mut wr).await);
    assert_eq!(n, 0);
}
