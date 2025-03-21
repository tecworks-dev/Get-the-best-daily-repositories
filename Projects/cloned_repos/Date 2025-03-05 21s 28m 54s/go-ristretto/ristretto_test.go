package ristretto_test

import (
	"bytes"
	"encoding/hex"
	"testing"

	"github.com/noxiousphot/go-ristretto"
)

func TestPointDerive(t *testing.T) {
	testVectors := []struct{ in, out string }{
		{"test", "b01d60504aa5f4c5bd9a7541c457661f9a789d18cb4e136e91d3c953488bd208"},
		{"pep", "3286c8d171dec02e70549c280d62524430408a781efc07e4428d1735671d195b"},
		{"ristretto", "c2f6bb4c4dab8feab66eab09e77e79b36095c86b3cd1145b9a2703205858d712"},
		{"elligator", "784c727b1e8099eb94e5a8edbd260363567fdbd35106a7a29c8b809cd108b322"},
	}
	for _, v := range testVectors {
		var p ristretto.Point
		p.Derive([]byte(v.in))
		out2 := hex.EncodeToString(p.Bytes())
		if out2 != v.out {
			t.Fatalf("Derive(%v) = %v != %v", v.in, v.out, out2)
		}
	}
}

// Test vectors from https://ristretto.group/test_vectors/ristretto255.html
func TestRistretto255TestVectors(t *testing.T) {
	smallMultiples := []string{
		"0000000000000000000000000000000000000000000000000000000000000000",
		"e2f2ae0a6abc4e71a884a961c500515f58e30b6aa582dd8db6a65945e08d2d76",
		"6a493210f7499cd17fecb510ae0cea23a110e8d5b901f8acadd3095c73a3b919",
		"94741f5d5d52755ece4f23f044ee27d5d1ea1e2bd196b462166b16152a9d0259",
		"da80862773358b466ffadfe0b3293ab3d9fd53c5ea6c955358f568322daf6a57",
		"e882b131016b52c1d3337080187cf768423efccbb517bb495ab812c4160ff44e",
		"f64746d3c92b13050ed8d80236a7f0007c3b3f962f5ba793d19a601ebb1df403",
		"44f53520926ec81fbd5a387845beb7df85a96a24ece18738bdcfa6a7822a176d",
		"903293d8f2287ebe10e2374dc1a53e0bc887e592699f02d077d5263cdd55601c",
		"02622ace8f7303a31cafc63f8fc48fdc16e1c8c8d234b2f0d6685282a9076031",
		"20706fd788b2720a1ed2a5dad4952b01f413bcf0e7564de8cdc816689e2db95f",
		"bce83f8ba5dd2fa572864c24ba1810f9522bc6004afe95877ac73241cafdab42",
		"e4549ee16b9aa03099ca208c67adafcafa4c3f3e4e5303de6026e3ca8ff84460",
		"aa52e000df2e16f55fb1032fc33bc42742dad6bd5a8fc0be0167436c5948501f",
		"46376b80f409b29dc2b5f6f0c52591990896e5716f41477cd30085ab7f10301e",
		"e0c418f7c8d9c4cdd7395b93ea124f3ad99021bb681dfc3302a9d99a2e53e64e",
	}
	var B, pt ristretto.Point
	B.SetBase()
	pt.SetZero()
	for i, pt2 := range smallMultiples {
		if hex.EncodeToString(pt.Bytes()) != pt2 {
			t.Fatalf("%d * B = %s != %v", i, pt2, hex.EncodeToString(pt.Bytes()))
		}
		pt.Add(&B, &pt)
	}

	badEncodings := []string{
		// Non-canonical field encodings.
		"00ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
		"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
		"f3ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
		"edffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",

		// Negative field elements.
		"0100000000000000000000000000000000000000000000000000000000000000",
		"01ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
		"ed57ffd8c914fb201471d1c3d245ce3c746fcbe63a3679d51b6a516ebebe0e20",
		"c34c4e1826e5d403b78e246e88aa051c36ccf0aafebffe137d148a2bf9104562",
		"c940e5a4404157cfb1628b108db051a8d439e1a421394ec4ebccb9ec92a8ac78",
		"47cfc5497c53dc8e61c91d17fd626ffb1c49e2bca94eed052281b510b1117a24",
		"f1c6165d33367351b0da8f6e4511010c68174a03b6581212c71c0e1d026c3c72",
		"87260f7a2f12495118360f02c26a470f450dadf34a413d21042b43b9d93e1309",

		// Non-square x^2.
		"26948d35ca62e643e26a83177332e6b6afeb9d08e4268b650f1f5bbd8d81d371",
		"4eac077a713c57b4f4397629a4145982c661f48044dd3f96427d40b147d9742f",
		"de6a7b00deadc788eb6b6c8d20c0ae96c2f2019078fa604fee5b87d6e989ad7b",
		"bcab477be20861e01e4a0e295284146a510150d9817763caf1a6f4b422d67042",
		"2a292df7e32cababbd9de088d1d1abec9fc0440f637ed2fba145094dc14bea08",
		"f4a9e534fc0d216c44b218fa0c42d99635a0127ee2e53c712f70609649fdff22",
		"8268436f8c4126196cf64b3c7ddbda90746a378625f9813dd9b8457077256731",
		"2810e5cbc2cc4d4eece54f61c6f69758e289aa7ab440b3cbeaa21995c2f4232b",

		// Negative xy value.
		"3eb858e78f5a7254d8c9731174a94f76755fd3941c0ac93735c07ba14579630e",
		"a45fdc55c76448c049a1ab33f17023edfb2be3581e9c7aade8a6125215e04220",
		"d483fe813c6ba647ebbfd3ec41adca1c6130c2beeee9d9bf065c8d151c5f396e",
		"8a2e1d30050198c65a54483123960ccc38aef6848e1ec8f5f780e8523769ba32",
		"32888462f8b486c68ad7dd9610be5192bbeaf3b443951ac1a8118419d9fa097b",
		"227142501b9d4355ccba290404bde41575b037693cef1f438c47f8fbf35d1165",
		"5c37cc491da847cfeb9281d407efc41e15144c876e0170b499a96a22ed31e01e",
		"445425117cb8c90edcbc7c1cc0e74f747f2c1efa5630a967c64f287792a48a4b",

		// s = -1, which causes y = 0.
		"ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
	}

	for _, ptHex := range badEncodings {
		var pt ristretto.Point
		var buf [32]byte
		tmp, _ := hex.DecodeString(ptHex)
		copy(buf[:], tmp)
		if pt.SetBytes(&buf) {
			t.Fatalf("%v should not decode", ptHex)
		}
	}

	encodedHashToPoints := []struct{ label, encoding string }{
		{"Ristretto is traditionally a short shot of espresso coffee",
			"3066f82a1a747d45120d1740f14358531a8f04bbffe6a819f86dfe50f44a0a46"},
		{"made with the normal amount of ground coffee but extracted with",
			"f26e5b6f7d362d2d2a94c5d0e7602cb4773c95a2e5c31a64f133189fa76ed61b"},
		{"about half the amount of water in the same amount of time",
			"006ccd2a9e6867e6a2c5cea83d3302cc9de128dd2a9a57dd8ee7b9d7ffe02826"},
		{"by using a finer grind.",
			"f8f0c87cf237953c5890aec3998169005dae3eca1fbb04548c635953c817f92a"},
		{"This produces a concentrated shot of coffee per volume.",
			"ae81e7dedf20a497e10c304a765c1767a42d6e06029758d2d7e8ef7cc4c41179"},
		{"Just pulling a normal shot short will produce a weaker shot",
			"e2705652ff9f5e44d3e841bf1c251cf7dddb77d140870d1ab2ed64f1a9ce8628"},
		{"and is not a Ristretto as some believe.",
			"80bd07262511cdde4863f8a7434cef696750681cb9510eea557088f76d9e5065"},
	}

	for _, tp := range encodedHashToPoints {
		var p ristretto.Point
		p.DeriveDalek([]byte(tp.label))

		res := hex.EncodeToString(p.Bytes())
		if res != tp.encoding {
			t.Fatalf("Test string %v produced %s instead of %s",
				tp.label, res, tp.encoding)
		}
	}
}

// Test base point multiplication
func TestBasePointMultiples(t *testing.T) {
	var s ristretto.Scalar
	var p1, p2, B ristretto.Point
	B.SetBase()
	for i := 0; i < 1000; i++ {
		s.Rand()
		p1.ScalarMultBase(&s)
		p2.ScalarMult(&B, &s)
		if !p1.Equals(&p2) {
			t.Fatalf("[%v]B = %v != %v", s, p2, p1)
		}
	}
}

func TestConditionalSet(t *testing.T) {
	var p1, p2 ristretto.Point
	for i := 0; i < 1000; i++ {
		p1.Rand()
		p2.Rand()
		p1.ConditionalSet(&p2, 0)
		if p1.Equals(&p2) {
			t.Fatal()
		}
		p1.ConditionalSet(&p2, 1)
		if !p1.Equals(&p2) {
			t.Fatal()
		}
	}
}

func testLizardVector(t *testing.T, in, out string) {
	var p ristretto.Point
	var inBuf [16]byte
	var outBuf [32]byte
	hex.Decode(inBuf[:], []byte(in))
	hex.Decode(outBuf[:], []byte(out))
	p.SetLizard(&inBuf)
	if !bytes.Equal(p.Bytes(), outBuf[:]) {
		t.Fatalf("Lizard(%s) is wrong", in)
	}
}

func TestLizardVectors(t *testing.T) {
	testLizardVector(t, "00000000000000000000000000000000",
		"f0b7e34484f74cf00f15024b738539738646bbbe1e9bc7509a676815227e774f")
	testLizardVector(t, "01010101010101010101010101010101",
		"cc92e81f585afc5caac88660d8d17e9025a44489a363042123f6af0702156e65")
}

func TestLizardInjective(t *testing.T) {
	var buf1, buf2 [16]byte
	var buf [32]byte
	var p ristretto.Point
	for i := 0; i < 1000; i++ {
		rnd.Read(buf1[:])
		p.SetLizard(&buf1)
		p.BytesInto(&buf)
		if !p.SetBytes(&buf) {
			t.Fatal()
		}
		err := p.LizardInto(&buf2)
		if err != nil {
			t.Fatalf("LizardInto: %v", err)
		}
		if !bytes.Equal(buf1[:], buf2[:]) {
			t.Fatalf("Lizard^-1 o Lizard != id: %v != %v", buf1, buf2)
		}
	}
}

func BenchmarkLizardDecode(b *testing.B) {
	buf := [16]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	var p ristretto.Point
	p.SetLizard(&buf)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		p.Lizard()
	}
}

func BenchmarkLizardEncode(b *testing.B) {
	var buf [16]byte
	var p ristretto.Point
	for n := 0; n < b.N; n++ {
		rnd.Read(buf[:])
		p.SetLizard(&buf)
	}
}
