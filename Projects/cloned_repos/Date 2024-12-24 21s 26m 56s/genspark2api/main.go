package main

import (
	"fmt"
	"genspark2api/check"
	"genspark2api/common"
	"genspark2api/common/config"
	logger "genspark2api/common/loggger"
	"genspark2api/middleware"
	"genspark2api/router"
	"github.com/gin-gonic/gin"
	"os"
	"strconv"
)

func main() {
	logger.SetupLogger()
	logger.SysLog(fmt.Sprintf("genspark2api %s started", common.Version))

	check.CheckEnvVariable()

	if os.Getenv("GIN_MODE") != "debug" {
		gin.SetMode(gin.ReleaseMode)
	}
	if config.DebugEnabled {
		logger.SysLog("running in debug mode")
	}
	var err error

	server := gin.New()
	server.Use(gin.Recovery())
	server.Use(middleware.RequestId())
	middleware.SetUpLogger(server)

	router.SetRouter(server)
	var port = os.Getenv("PORT")
	if port == "" {
		port = strconv.Itoa(*common.Port)
	}
	err = server.Run(":" + port)
	if err != nil {
		logger.FatalLog("failed to start HTTP server: " + err.Error())
	}
}
