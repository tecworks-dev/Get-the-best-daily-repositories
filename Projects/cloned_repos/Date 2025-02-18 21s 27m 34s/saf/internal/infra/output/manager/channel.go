package manager

import (
	"github.com/angelicmuklu/saf/internal/infra/output"
	"github.com/angelicmuklu/saf/internal/infra/output/mqtt"
	"github.com/angelicmuklu/saf/internal/infra/output/printer"
)

// list of available channles, please add each channel into this list to make them available.
func channels() []output.Channel {
	return []output.Channel{
		new(printer.Printer),
		new(mqtt.MQTT),
	}
}
