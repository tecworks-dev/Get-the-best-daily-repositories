/*
Copyright IBM Corp. All Rights Reserved.

SPDX-License-Identifier: Apache-2.0
*/
package types

import (
	"crypto"

	math "github.com/IBM/mathlib"
)

// RevocationAlgorithm identifies the revocation algorithm
type RevocationAlgorithm int32

const (
	// IDEMIX constant to identify Idemix related algorithms
	IDEMIX = "IDEMIX"
)

const (
	// AlgNoRevocation means no revocation support
	AlgNoRevocation RevocationAlgorithm = iota
)

// IdemixIssuerKeyGenOpts contains the options for the Idemix Issuer key-generation.
// A list of attribytes may be optionally passed
type IdemixIssuerKeyGenOpts struct {
	// Temporary tells if the key is ephemeral
	Temporary bool
	// AttributeNames is a list of attributes
	AttributeNames []string
}

// Algorithm returns the key generation algorithm identifier (to be used).
func (*IdemixIssuerKeyGenOpts) Algorithm() string {
	return IDEMIX
}

// Ephemeral returns true if the key to generate has to be ephemeral,
// false otherwise.
func (o *IdemixIssuerKeyGenOpts) Ephemeral() bool {
	return o.Temporary
}

// CommitmentBasesRequest describes the request for commitment bases
type CommitmentBasesRequest int

const (
	None CommitmentBasesRequest = iota
	Dlog
)

// CommitmentType describes a commitment performed by the scheme
type CommitmentType int

const (
	Nym CommitmentType = iota + 1
	NymEid
	NymRH
)

// IdemixIssuerPublicKeyImportOpts contains the options for importing of an Idemix issuer public key.
type IdemixIssuerPublicKeyImportOpts struct {
	Temporary bool
	// AttributeNames is a list of attributes to ensure the import public key has
	AttributeNames []string
	// CommitmentBasesRequest describes the request for commitment bases
	CommitmentBasesRequest CommitmentBasesRequest
	// RhIndex is the index of attribute containing the revocation handler.
	RhIndex int
	// EidIndex contains the index of the EID attribute
	EidIndex int
	// SKIndex contains the index of the secret key
	SKIndex int
	// CommitmentBases contains the bases used for the various commitments
	CommitmentBases map[CommitmentType]interface{}
}

// Algorithm returns the key generation algorithm identifier (to be used).
func (*IdemixIssuerPublicKeyImportOpts) Algorithm() string {
	return IDEMIX
}

// Ephemeral returns true if the key to generate has to be ephemeral,
// false otherwise.
func (o *IdemixIssuerPublicKeyImportOpts) Ephemeral() bool {
	return o.Temporary
}

// IdemixIssuerKeyImportOpts contains the options for importing of an Idemix issuer public key.
type IdemixIssuerKeyImportOpts struct {
	Temporary bool
	// AttributeNames is a list of attributes to ensure the import public key has
	AttributeNames []string
}

// Algorithm returns the key generation algorithm identifier (to be used).
func (*IdemixIssuerKeyImportOpts) Algorithm() string {
	return IDEMIX
}

// Ephemeral returns true if the key to generate has to be ephemeral,
// false otherwise.
func (o *IdemixIssuerKeyImportOpts) Ephemeral() bool {
	return o.Temporary
}

// IdemixUserSecretKeyGenOpts contains the options for the generation of an Idemix credential secret key.
type IdemixUserSecretKeyGenOpts struct {
	Temporary bool
}

// Algorithm returns the key generation algorithm identifier (to be used).
func (*IdemixUserSecretKeyGenOpts) Algorithm() string {
	return IDEMIX
}

// Ephemeral returns true if the key to generate has to be ephemeral,
// false otherwise.
func (o *IdemixUserSecretKeyGenOpts) Ephemeral() bool {
	return o.Temporary
}

// IdemixUserSecretKeyImportOpts contains the options for importing of an Idemix credential secret key.
type IdemixUserSecretKeyImportOpts struct {
	Temporary bool
}

// Algorithm returns the key generation algorithm identifier (to be used).
func (*IdemixUserSecretKeyImportOpts) Algorithm() string {
	return IDEMIX
}

// Ephemeral returns true if the key to generate has to be ephemeral,
// false otherwise.
func (o *IdemixUserSecretKeyImportOpts) Ephemeral() bool {
	return o.Temporary
}

// IdemixNymKeyDerivationOpts contains the options to create a new unlinkable pseudonym from a
// credential secret key with the respect to the specified issuer public key
type IdemixNymKeyDerivationOpts struct {
	// Temporary tells if the key is ephemeral
	Temporary bool
	// IssuerPK is the public-key of the issuer
	IssuerPK Key
}

// Algorithm returns the key derivation algorithm identifier (to be used).
func (*IdemixNymKeyDerivationOpts) Algorithm() string {
	return IDEMIX
}

// Ephemeral returns true if the key to derive has to be ephemeral,
// false otherwise.
func (o *IdemixNymKeyDerivationOpts) Ephemeral() bool {
	return o.Temporary
}

// IssuerPublicKey returns the issuer public key used to derive
// a new unlinkable pseudonym from a credential secret key
func (o *IdemixNymKeyDerivationOpts) IssuerPublicKey() Key {
	return o.IssuerPK
}

// IdemixNymPublicKeyImportOpts contains the options to import the public part of a pseudonym
type IdemixNymPublicKeyImportOpts struct {
	// Temporary tells if the key is ephemeral
	Temporary bool
}

// Algorithm returns the key derivation algorithm identifier (to be used).
func (*IdemixNymPublicKeyImportOpts) Algorithm() string {
	return IDEMIX
}

// Ephemeral returns true if the key to derive has to be ephemeral,
// false otherwise.
func (o *IdemixNymPublicKeyImportOpts) Ephemeral() bool {
	return o.Temporary
}

// IdemixNymKeyImportOpts contains the options to import a pseudonym
type IdemixNymKeyImportOpts struct {
	// Temporary tells if the key is ephemeral
	Temporary bool
}

// Algorithm returns the key derivation algorithm identifier (to be used).
func (*IdemixNymKeyImportOpts) Algorithm() string {
	return IDEMIX
}

// Ephemeral returns true if the key to derive has to be ephemeral,
// false otherwise.
func (o *IdemixNymKeyImportOpts) Ephemeral() bool {
	return o.Temporary
}

// IdemixCredentialRequestSignerOpts contains the option to create a Idemix credential request.
type IdemixCredentialRequestSignerOpts struct {
	// Attributes contains a list of indices of the attributes to be included in the
	// credential. The indices are with the respect to IdemixIssuerKeyGenOpts#AttributeNames.
	Attributes []int
	// IssuerPK is the public-key of the issuer
	IssuerPK Key
	// IssuerNonce is generated by the issuer and used by the client to generate the credential request.
	// Once the issuer gets the credential requests, it checks that the nonce is the same.
	IssuerNonce []byte
	// HashFun is the hash function to be used
	H crypto.Hash
}

func (o *IdemixCredentialRequestSignerOpts) HashFunc() crypto.Hash {
	return o.H
}

// IdemixCredentialRequestSignerOpts contains the option to create a Idemix credential request.
type IdemixBlindCredentialRequestSignerOpts struct {
	// Attributes contains a list of indices of the attributes to be included in the
	// credential. The indices are with the respect to IdemixIssuerKeyGenOpts#AttributeNames.
	Attributes []int
	// IssuerPK is the public-key of the issuer
	IssuerPK Key
	// IssuerNonce is generated by the issuer and used by the client to generate the credential request.
	// Once the issuer gets the credential requests, it checks that the nonce is the same.
	IssuerNonce []byte
	// HashFun is the hash function to be used
	H crypto.Hash

	// Blinding contains the blinding used to mask the unrevealed attributes
	Blinding []byte
}

func (o *IdemixBlindCredentialRequestSignerOpts) HashFunc() crypto.Hash {
	return o.H
}

// IssuerPublicKey returns the issuer public key used to derive
// a new unlinkable pseudonym from a credential secret key
func (o *IdemixCredentialRequestSignerOpts) IssuerPublicKey() Key {
	return o.IssuerPK
}

// IdemixAttributeType represents the type of an idemix attribute
type IdemixAttributeType int

const (
	// IdemixHiddenAttribute represents an hidden attribute
	IdemixHiddenAttribute IdemixAttributeType = iota
	// IdemixStringAttribute represents a sequence of bytes
	IdemixBytesAttribute
	// IdemixIntAttribute represents an int
	IdemixIntAttribute
)

type IdemixAttribute struct {
	// Type is the attribute's type
	Type IdemixAttributeType
	// Value is the attribute's value
	Value interface{}
}

// IdemixCredentialSignerOpts contains the options to produce a credential starting from a credential request
type IdemixCredentialSignerOpts struct {
	// Attributes to include in the credentials. IdemixHiddenAttribute is not allowed here
	Attributes []IdemixAttribute
	// IssuerPK is the public-key of the issuer
	IssuerPK Key
	// HashFun is the hash function to be used
	H crypto.Hash
}

// HashFunc returns an identifier for the hash function used to produce
// the message passed to Signer.Sign, or else zero to indicate that no
// hashing was done.
func (o *IdemixCredentialSignerOpts) HashFunc() crypto.Hash {
	return o.H
}

func (o *IdemixCredentialSignerOpts) IssuerPublicKey() Key {
	return o.IssuerPK
}

// AttrNymAuditData contains the data that is used to audit a commitment to an auditable attribute.
// Nym is a commitment to the Attr. Attr can be an Enrollment ID or a Revocation Handle.
// Notice that this data should be used only after validating the corresponding signature.
type AttrNymAuditData struct {
	// Nym is the commitment to an attribute
	Nym *math.G1

	// RAttrNym is the randomness used to generate the Attr Nym
	Rand *math.Zr

	// Attr is the attribute (enrollment id or revocation handle)
	Attr *math.Zr
}

type IdemixSignerMetadata struct {
	EidNym          []byte
	EidNymAuditData *AttrNymAuditData
	RhNym           []byte
	RhNymAuditData  *AttrNymAuditData
}

// IdemixSignerOpts contains the options to generate an Idemix signature
type IdemixSignerOpts struct {
	// Nym is the pseudonym to be used
	Nym Key
	// IssuerPK is the public-key of the issuer
	IssuerPK Key
	// Credential is the byte representation of the credential signed by the issuer
	Credential []byte
	// Attributes specifies which attribute should be disclosed and which not.
	// If Attributes[i].Type = IdemixHiddenAttribute
	// then the i-th credential attribute should not be disclosed, otherwise the i-th
	// credential attribute will be disclosed.
	// At verification time, if the i-th attribute is disclosed (Attributes[i].Type != IdemixHiddenAttribute),
	// then Attributes[i].Value must be set accordingly.
	Attributes []IdemixAttribute
	// RhIndex is the index of attribute containing the revocation handler.
	// Notice that this attributed cannot be disclosed
	RhIndex int
	// EidIndex contains the index of the EID attribute
	EidIndex int
	// SKIndex contains the index of the secret key
	SKIndex int
	// CRI contains the credential revocation information
	CRI []byte
	// Epoch is the revocation epoch the signature should be produced against
	Epoch int
	// RevocationPublicKey is the revocation public key
	RevocationPublicKey Key
	// H is the hash function to be used
	H crypto.Hash
	// SigType is the type of signature that shall be generated
	SigType SignatureType
	// IdemixSignerMetadata contains metadata about the signature
	Metadata *IdemixSignerMetadata
	// VerificationType controls what type of verification the caller expects
	VerificationType VerificationType
}

func (o *IdemixSignerOpts) HashFunc() crypto.Hash {
	return o.H
}

// EidNymAuditOpts contains audit options for pseudonymous enrollment id
type EidNymAuditOpts struct {
	AuditVerificationType AuditVerificationType
	EidIndex              int      // Index of enrollment ID attribute in signature
	EnrollmentID          string   // Enrollment ID of identity
	RNymEid               *math.Zr // Field element of randomness
	SKIndex               int      // index of the secret key
}

func (o *EidNymAuditOpts) HashFunc() crypto.Hash {
	return 0
}

// RhNymAuditOpts contains audit options for pseudonymous revocation handle
type RhNymAuditOpts struct {
	AuditVerificationType AuditVerificationType
	RhIndex               int      // Index of revocation handle attribute in signature
	RevocationHandle      string   // Revocation handle of identity
	RNymRh                *math.Zr // Field element of randomness
	SKIndex               int      // index of the secret key
}

func (o *RhNymAuditOpts) HashFunc() crypto.Hash {
	return 0
}

// IdemixNymSignerOpts contains the options to generate an idemix pseudonym signature.
type IdemixNymSignerOpts struct {
	// Nym is the pseudonym to be used
	Nym Key
	// IssuerPK is the public-key of the issuer
	IssuerPK Key
	// H is the hash function to be used
	H crypto.Hash

	// IsSmartcard checks whether this nym signature was generated by a smartcard
	IsSmartcard bool

	// Smartcard is a software smartcard used to support signing in s/w
	// this field is only used in testing to emulate a real smartcard
	// and may never be set in production.
	Smartcard interface{}

	// NymEid is the nym eid to use in the verification
	NymEid *math.G1

	// data about the nym returned by the smartcard implementation
	NymG1 *math.G1
	RNym  *math.Zr

	SKIndex int // index of the secret key
}

// HashFunc returns an identifier for the hash function used to produce
// the message passed to Signer.Sign, or else zero to indicate that no
// hashing was done.
func (o *IdemixNymSignerOpts) HashFunc() crypto.Hash {
	return o.H
}

// IdemixRevocationKeyGenOpts contains the options for the Idemix revocation key-generation.
type IdemixRevocationKeyGenOpts struct {
	// Temporary tells if the key is ephemeral
	Temporary bool
}

// Algorithm returns the key generation algorithm identifier (to be used).
func (*IdemixRevocationKeyGenOpts) Algorithm() string {
	return IDEMIX
}

// Ephemeral returns true if the key to generate has to be ephemeral,
// false otherwise.
func (o *IdemixRevocationKeyGenOpts) Ephemeral() bool {
	return o.Temporary
}

// IdemixRevocationPublicKeyImportOpts contains the options for importing of an Idemix revocation public key.
type IdemixRevocationPublicKeyImportOpts struct {
	Temporary bool
}

// Algorithm returns the key generation algorithm identifier (to be used).
func (*IdemixRevocationPublicKeyImportOpts) Algorithm() string {
	return IDEMIX
}

// Ephemeral returns true if the key to generate has to be ephemeral,
// false otherwise.
func (o *IdemixRevocationPublicKeyImportOpts) Ephemeral() bool {
	return o.Temporary
}

// IdemixRevocationKeyImportOpts contains the options for importing of an Idemix revocation key pair.
type IdemixRevocationKeyImportOpts struct {
	Temporary bool
}

// Algorithm returns the key generation algorithm identifier (to be used).
func (*IdemixRevocationKeyImportOpts) Algorithm() string {
	return IDEMIX
}

// Ephemeral returns true if the key to generate has to be ephemeral,
// false otherwise.
func (o *IdemixRevocationKeyImportOpts) Ephemeral() bool {
	return o.Temporary
}

// IdemixCRISignerOpts contains the options to generate an Idemix CRI.
// The CRI is supposed to be generated by the Issuing authority and
// can be verified publicly by using the revocation public key.
type IdemixCRISignerOpts struct {
	Epoch               int
	RevocationAlgorithm RevocationAlgorithm
	UnrevokedHandles    [][]byte
	// H is the hash function to be used
	H crypto.Hash
}

func (o *IdemixCRISignerOpts) HashFunc() crypto.Hash {
	return o.H
}

// SignatureType describes the type of idemix signature
type SignatureType int

const (
	// Standard is the base signature type
	Standard SignatureType = iota
	// EidNym adds a hiding and binding commitment to the enrollment id and proves its correctness
	EidNym
	// EidNymRhNym adds a hiding and binding commitment to both the enrollment id and the revocation handle and proves each of their correctness
	EidNymRhNym
	// Smartcard has a separate component do the nym signing
	Smartcard
	// SmartcardNoNyms has a separate component do the nym signing and has no extra commitments
	SmartcardNoNyms
)

// VerificationType describes the type of verification that is required
type VerificationType int

const (
	// Basic performs the verification without any of the extensions (e.g. it ignores the eid nym and rh nym)
	Basic VerificationType = iota
	// BestEffort performs all verifications possible given the available information in the signature/opts
	BestEffort
	// ExpectStandard expects a SignatureType of type Standard
	ExpectStandard
	// ExpectEidNym expects a SignatureType of type EidNym
	ExpectEidNym
	// ExpectEidNymRhNym expects a SignatureType of EidNymRhNym
	ExpectEidNymRhNym
	// ExpectSmartcard has a separate component do the nym signing
	ExpectSmartcard
	// ExpectSmartcardNoNyms has a separate component do the nym signing and has no extra commitments
	ExpectSmartcardNoNyms
)

// AuditVerificationType describes the type of audit verification that is required
type AuditVerificationType int

const (
	// AuditExpectSignature performs the audit verification against a signature
	AuditExpectSignature AuditVerificationType = iota
	// AuditExpectEidNym performs the audit verification against an EID pseudonym
	AuditExpectEidNym
	// AuditExpectEidNymRhNym performs the audit verification against an EID pseudonym and a Revocation Handle pseudonym
	AuditExpectEidNymRhNym
)
