package handler_test

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/angelicmuklu/saf/internal/infra/cmq"
	"github.com/angelicmuklu/saf/internal/infra/config"
	"github.com/angelicmuklu/saf/internal/infra/http/handler"
	"github.com/angelicmuklu/saf/internal/infra/http/request"
	"github.com/angelicmuklu/saf/internal/infra/logger"
	"github.com/angelicmuklu/saf/internal/infra/telemetry"
	"github.com/gofiber/fiber/v2"
	"github.com/stretchr/testify/suite"
	"go.opentelemetry.io/otel"
	"go.uber.org/fx"
	"go.uber.org/fx/fxtest"
	"go.uber.org/zap"
)

type EventSuite struct {
	suite.Suite

	engine *fiber.App
	app    *fxtest.App
}

func (suite *EventSuite) SetupSuite() {
	suite.app = fxtest.New(suite.T(),
		fx.Provide(config.Provide),
		fx.Provide(logger.Provide),
		fx.Provide(telemetry.ProvideNull),
		fx.Provide(cmq.Provide),
		fx.Invoke(func(cmq *cmq.CMQ, logger *zap.Logger, _ telemetry.Telemetery) {
			suite.Require().NoError(cmq.Streams(context.Background()))
			suite.engine = fiber.New()

			handler.Event{
				CMQ:    cmq,
				Logger: logger,
				Tracer: otel.GetTracerProvider().Tracer(""),
			}.Register(suite.engine.Group(""))
		}),
	).RequireStart()
}

func (suite *EventSuite) TestHandler() {
	require := suite.Require()

	payload, err := json.Marshal(request.Event{
		Subject: "hello",
		ID:      "",
		Service: "OfferService",
		Payload: []byte("from the otherside"),
	})
	require.NoError(err)

	req := httptest.NewRequest(http.MethodPost, "/event", bytes.NewBuffer(payload))
	req.Header.Set(fiber.HeaderContentType, fiber.MIMEApplicationJSON)

	resp, err := suite.engine.Test(req)
	require.NoError(err)

	body, err := io.ReadAll(resp.Body)
	require.NoError(err)

	require.Equal(http.StatusOK, resp.StatusCode, string(body))
	require.NoError(resp.Body.Close())
}

func TestEventSuite(t *testing.T) {
	t.Parallel()
	suite.Run(t, new(EventSuite))
}
