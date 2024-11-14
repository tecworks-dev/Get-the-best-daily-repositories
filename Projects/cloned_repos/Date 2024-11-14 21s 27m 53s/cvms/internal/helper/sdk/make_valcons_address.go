package sdkhelper

import (
	"encoding/base64"
	"encoding/hex"
)

// NOTE: only support ed25519 key type for ICS
func MakeValconsAddressFromPubeky(pubkey string, hrp string) (
	/* valcons address */ string,
	/* unexpected err */ error,
) {
	decodedKey, err := base64.StdEncoding.DecodeString(pubkey)
	if err != nil {
		return "", err
	}
	hexAddress, err := MakeProposerAddress(Ed25519, decodedKey)
	if err != nil {
		return "", err
	}
	bz, err := hex.DecodeString(hexAddress)
	if err != nil {
		return "", err
	}
	valconsAddress, err := ConvertAndEncode(hrp, bz)
	if err != nil {
		return "", err
	}
	return valconsAddress, nil
}
