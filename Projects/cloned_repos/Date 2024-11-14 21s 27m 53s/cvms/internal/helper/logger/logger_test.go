package logger

import (
	"strings"
	"testing"
	"time"

	"github.com/sirupsen/logrus"
)

func TestLoggerPrint(t *testing.T) {
	logrus.StandardLogger().Println("test start")

	testLogger, _ := GetLogger("false", "5")

	testEntry := testLogger.WithFields(logrus.Fields{
		"package":       "walrus",
		"instance_name": 10,
		"test":          "test",
	})

	testEntry.Infoln("test")
	testEntry.Errorln("test")
	testEntry.Debugln("test")
	testEntry.Warnln("test")

	App := "testApp"
	sleep := 15 * time.Second

	testEntry.Infof("[%s] updated metrics successfully and going to sleep %s...", App, sleep.String())

}

func TestSortFunction(t *testing.T) {
	testFields := []string{"time", "level", "instance_name", "package", "msg"}
	// var fieldSeq = map[string]int{
	// 	"time":          0,
	// 	"level":         1,
	// 	"instance_name": 2,
	// 	"package":       3,
	// 	"msg":           4,
	// }
	sortingFunc(testFields)
}

func TestFileName(t *testing.T) {
	testString := "/Users/jeongseup/Workspace/validator-monitoring-service/packages/uptime/collector/collector.go"

	result := strings.SplitAfter(testString, WORKSPACE)
	t.Log(result[1])
}
