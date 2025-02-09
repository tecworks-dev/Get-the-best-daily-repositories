// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'store_transaction.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

_$StoreTransactionImpl _$$StoreTransactionImplFromJson(Map json) =>
    _$StoreTransactionImpl(
      _readTransactionIdentifier(json, 'transactionIdentifier') as String,
      _readTransactionIdentifier(json, 'wildberryIdentifier') as String,
      json['productIdentifier'] as String,
      json['purchaseDate'] as String,
    );

Map<String, dynamic> _$$StoreTransactionImplToJson(
        _$StoreTransactionImpl instance) =>
    <String, dynamic>{
      'transactionIdentifier': instance.transactionIdentifier,
      'wildberryIdentifier': instance.wildberryIdentifier,
      'productIdentifier': instance.productIdentifier,
      'purchaseDate': instance.purchaseDate,
    };
