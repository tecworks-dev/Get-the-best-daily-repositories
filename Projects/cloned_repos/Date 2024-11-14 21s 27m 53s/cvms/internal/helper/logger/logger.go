package logger

import (
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"

	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
)

const WORKSPACE = "cvms"

// field key names
const (
	FieldKeyChain   = "chain"
	FieldKeyChainID = "chain_id"
	FieldKeyPackage = "package"
)

func getLoggerRecover() {
	if r := recover(); r != nil {
		fmt.Println("Recovered", r)
		debug.PrintStack()
	}
}

// GetLogger returns the logger instance.
// This instance is the entry point for all logging
func GetLogger(logColorDisable string, logLevel string) (*logrus.Logger, error) {
	defer getLoggerRecover()

	logColorDisableValue, err := strconv.ParseBool(logColorDisable)
	if err != nil {
		return nil, errors.Wrap(err, "check your log-color-disable flag")
	}
	logLevelValue, err := strconv.Atoi(logLevel)
	if err != nil {
		return nil, errors.Wrap(err, "check your log-level flag")
	}

	// Get the current working directory dynamically
	rootPath, err := os.Getwd()
	if err != nil {
		return nil, errors.Wrap(err, "failed to get working directory")
	}

	// Normalize the rootPath to include trailing slash
	if !strings.HasSuffix(rootPath, "/") {
		rootPath += "/"
	}

	logger := logrus.New()
	logger.SetLevel(logrus.Level(logLevelValue))
	logger.SetOutput(os.Stdout)
	logger.SetFormatter(&logrus.TextFormatter{
		DisableQuote: false,
		ForceQuote:   true,
		// time
		DisableTimestamp: true,
		FullTimestamp:    false,
		// level
		DisableLevelTruncation: false,
		PadLevelText:           true,
		DisableColors:          logColorDisableValue,
		// sorting
		SortingFunc:    sortingFunc,
		DisableSorting: false,
		// field
		QuoteEmptyFields: true,
		FieldMap: logrus.FieldMap{
			logrus.FieldKeyLevel: "level",
			logrus.FieldKeyMsg:   "msg",
			logrus.FieldKeyFile:  "file",
			logrus.FieldKeyFunc:  "func",
			FieldKeyChain:        "chain",
			FieldKeyPackage:      "package",
		},

		// Modify the CallerPrettyfier to trim the dynamic rootPath
		CallerPrettyfier: func(f *runtime.Frame) (string, string) {
			// Trim the dynamic rootPath from the file path
			file := strings.TrimPrefix(f.File, rootPath)
			// Return the formatted file path and line number
			return "", fmt.Sprintf("%s:%d", file, f.Line)
		},
	})

	// Disable caller if not in debug mode
	if logLevelValue == 5 {
		logger.SetReportCaller(true)
	}

	logger.Debugf("logger's root path: %s", rootPath)
	return logger, nil
}

var fieldSeq = map[string]int{
	logrus.FieldKeyLevel: 1,
	logrus.FieldKeyMsg:   2,
	FieldKeyChain:        3,
	FieldKeyChainID:      4,
	FieldKeyPackage:      5,
	logrus.FieldKeyFile:  6,
}

func sortingFunc(fields []string) {
	sort.Slice(fields, func(i, j int) bool {
		if iIndex, iOk := fieldSeq[fields[i]]; iOk {
			if jIndex, jOk := fieldSeq[fields[j]]; jOk {
				return iIndex < jIndex
			}
			return true
		}
		return false
	})
}

func GetTestLogger() *logrus.Logger {
	logger := logrus.New()
	logger.SetLevel(logrus.Level(5))
	logger.SetOutput(os.Stdout)
	logger.SetReportCaller(true)
	logger.SetFormatter(&logrus.JSONFormatter{
		PrettyPrint: true,
	})

	return logger
}
