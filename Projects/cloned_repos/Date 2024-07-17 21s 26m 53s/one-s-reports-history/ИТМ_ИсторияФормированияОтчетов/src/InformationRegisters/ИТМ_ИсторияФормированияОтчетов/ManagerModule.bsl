// @strict-types

#Если Сервер Или ТолстыйКлиентОбычноеПриложение Или ВнешнееСоединение Тогда

#Область СлужебныйПрограммныйИнтерфейс

// Содержимое записи.
// 
// Параметры:
//  КлючЗаписиРегистра - РегистрСведенийКлючЗаписи.ИТМ_ИсторияФормированияОтчетов - Ключ записи регистра
//  ИзвлекатьХранилища - Булево - Извлекать хранилища
//  ИзвлекатьДополнительныеСвойства - Булево - Извлекать дополнительные свойства
// 
// Возвращаемое значение:
//  см. ИТМ_ИсторияФормированияОтчетовКлиентСервер.ШаблонЗаписиИсторииФормированияОтчетов
Функция СодержимоеЗаписи(КлючЗаписиРегистра, ИзвлекатьХранилища = Истина, ИзвлекатьДополнительныеСвойства = Ложь) Экспорт
	
	СодержимоеЗаписи	= ИТМ_ИсторияФормированияОтчетовКлиентСервер.ШаблонЗаписиИсторииФормированияОтчетов();
	
	ЗаписьРегистра	= СоздатьМенеджерЗаписи();
	ЗаполнитьЗначенияСвойств(ЗаписьРегистра, КлючЗаписиРегистра);
	ЗаписьРегистра.Прочитать();
	
	ИменаКоллекций   = "Измерения,Ресурсы,Реквизиты";
	КоллекцияСвойств = СтрРазделить(ИменаКоллекций, ",");
	МетаданныеРегистра = Метаданные.РегистрыСведений.ИТМ_ИсторияФормированияОтчетов;
	Для Каждого ТекущаяКоллекцияСвойств Из КоллекцияСвойств Цикл
		Для Каждого ТекущееСвойство Из МетаданныеРегистра[ТекущаяКоллекцияСвойств] Цикл //ОбъектМетаданных
			
			ЗначениеТекущегоСвойства = ЗаписьРегистра[ТекущееСвойство.Имя]; //Произвольный
			
			Если ТипЗнч(ЗначениеТекущегоСвойства) = Тип("ХранилищеЗначения") Тогда
				ИзвлечьСодержимое = ИзвлекатьХранилища 
					И (ИзвлекатьДополнительныеСвойства ИЛИ ТекущееСвойство.Имя <> "ДополнительныеСвойства");
				Если ИзвлечьСодержимое Тогда
					ЗначениеТекущегоСвойства = ЗначениеТекущегоСвойства.Получить(); //Произвольный, ХранилищеЗначения - ЕДТ хочет
				КонецЕсли;
			КонецЕсли;
			
			СодержимоеЗаписи.Вставить(ТекущееСвойство.Имя, ЗначениеТекущегоСвойства);
			
		КонецЦикла;
	КонецЦикла;
	
	Возврат СодержимоеЗаписи;
		
КонецФункции

// Записать данные записи.
// 
// Параметры:
//  ДанныеЗаписи - Структура - Данные записи
//  ПомещатьВХранилище - Булево - Помещать в хранилище
Процедура ЗаписатьДанныеЗаписи(ДанныеЗаписи, ПомещатьВХранилище = Истина) Экспорт
	
	ЗаписьРегистра	= СоздатьМенеджерЗаписи();
	МетаданныеРегистра = Метаданные.РегистрыСведений.ИТМ_ИсторияФормированияОтчетов;
	
	ИменаКоллекций   = "Измерения,Ресурсы,Реквизиты";
	КоллекцияСвойств = СтрРазделить(ИменаКоллекций, ",");
	Для Каждого ТекущаяКоллекцияСвойств Из КоллекцияСвойств Цикл
		Для Каждого ТекущееСвойство Из МетаданныеРегистра[ТекущаяКоллекцияСвойств] Цикл //ОбъектМетаданныхРеквизит
			
			ЗначениеТекущегоСвойства = Неопределено;
			Если НЕ ДанныеЗаписи.Свойство(ТекущееСвойство.Имя, ЗначениеТекущегоСвойства) Тогда
				Продолжить;
			КонецЕсли;
			
			Если ПомещатьВХранилище И ТекущееСвойство.Тип.СодержитТип(Тип("ХранилищеЗначения")) Тогда
				ЗначениеТекущегоСвойства = Новый ХранилищеЗначения(ЗначениеТекущегоСвойства);
			КонецЕсли;
			
			ЗаписьРегистра[ТекущееСвойство.Имя] = ЗначениеТекущегоСвойства;
			
		КонецЦикла;
	КонецЦикла;
	
	ЗаписьРегистра.Записать(Истина);
	
КонецПроцедуры

#КонецОбласти

#КонецЕсли
