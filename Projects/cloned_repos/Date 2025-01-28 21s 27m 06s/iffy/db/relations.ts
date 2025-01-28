import { relations } from "drizzle-orm/relations";
import {
  records,
  moderations,
  rulesets,
  recordUsers,
  recordUserActions,
  appeals,
  messages,
  appealActions,
  moderationsToRules,
  webhookEndpoints,
  webhookEvents,
  rules,
  presets,
  ruleStrategies,
  presetStrategies,
} from "./tables";

export const moderationsRelations = relations(moderations, ({ one, many }) => ({
  record: one(records, {
    fields: [moderations.recordId],
    references: [records.id],
  }),
  ruleset: one(rulesets, {
    fields: [moderations.rulesetId],
    references: [rulesets.id],
  }),
  moderationsToRules: many(moderationsToRules),
}));

export const recordsRelations = relations(records, ({ one, many }) => ({
  moderations: many(moderations),
  recordUser: one(recordUsers, {
    fields: [records.recordUserId],
    references: [recordUsers.id],
  }),
}));

export const rulesetsRelations = relations(rulesets, ({ many }) => ({
  moderations: many(moderations),
  rules: many(rules),
}));

export const recordUserActionsRelations = relations(recordUserActions, ({ one, many }) => ({
  recordUser: one(recordUsers, {
    fields: [recordUserActions.recordUserId],
    references: [recordUsers.id],
  }),
  messages: many(messages),
  appeal: one(appeals),
}));

export const recordUsersRelations = relations(recordUsers, ({ many }) => ({
  actions: many(recordUserActions),
  from: many(messages, { relationName: "from" }),
  to: many(messages, { relationName: "to" }),
  records: many(records),
}));

export const messagesRelations = relations(messages, ({ one }) => ({
  appeal: one(appeals, {
    fields: [messages.appealId],
    references: [appeals.id],
  }),
  from: one(recordUsers, {
    fields: [messages.fromId],
    references: [recordUsers.id],
    relationName: "from",
  }),
  recordUserAction: one(recordUserActions, {
    fields: [messages.recordUserActionId],
    references: [recordUserActions.id],
  }),
  to: one(recordUsers, {
    fields: [messages.toId],
    references: [recordUsers.id],
    relationName: "to",
  }),
}));

export const appealsRelations = relations(appeals, ({ one, many }) => ({
  messages: many(messages),
  recordUserAction: one(recordUserActions, {
    fields: [appeals.recordUserActionId],
    references: [recordUserActions.id],
  }),
  actions: many(appealActions),
}));

export const appealActionsRelations = relations(appealActions, ({ one }) => ({
  appeal: one(appeals, {
    fields: [appealActions.appealId],
    references: [appeals.id],
  }),
}));

export const moderationsToRulesRelations = relations(moderationsToRules, ({ one }) => ({
  moderation: one(moderations, {
    fields: [moderationsToRules.moderationId],
    references: [moderations.id],
  }),
  rule: one(rules, {
    fields: [moderationsToRules.ruleId],
    references: [rules.id],
  }),
}));

export const webhookEventsRelations = relations(webhookEvents, ({ one }) => ({
  webhookEndpoint: one(webhookEndpoints, {
    fields: [webhookEvents.webhookEndpointId],
    references: [webhookEndpoints.id],
  }),
}));

export const webhookEndpointsRelations = relations(webhookEndpoints, ({ many }) => ({
  webhookEvents: many(webhookEvents),
}));

export const rulesRelations = relations(rules, ({ one, many }) => ({
  ruleset: one(rulesets, {
    fields: [rules.rulesetId],
    references: [rulesets.id],
  }),
  preset: one(presets, {
    fields: [rules.presetId],
    references: [presets.id],
  }),
  strategies: many(ruleStrategies),
  moderationsToRules: many(moderationsToRules),
}));

export const presetsRelations = relations(presets, ({ many }) => ({
  strategies: many(presetStrategies),
  rules: many(rules),
}));

export const ruleStrategiesRelations = relations(ruleStrategies, ({ one }) => ({
  rule: one(rules, {
    fields: [ruleStrategies.ruleId],
    references: [rules.id],
  }),
}));

export const presetStrategiesRelations = relations(presetStrategies, ({ one }) => ({
  preset: one(presets, {
    fields: [presetStrategies.presetId],
    references: [presets.id],
  }),
}));
