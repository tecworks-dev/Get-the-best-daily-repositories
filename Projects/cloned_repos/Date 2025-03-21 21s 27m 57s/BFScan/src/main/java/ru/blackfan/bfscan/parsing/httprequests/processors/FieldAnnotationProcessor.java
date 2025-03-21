package ru.blackfan.bfscan.parsing.httprequests.processors;

import jadx.api.plugins.input.data.annotations.EncodedType;
import jadx.api.plugins.input.data.annotations.EncodedValue;
import jadx.api.plugins.input.data.annotations.IAnnotation;
import jadx.api.plugins.input.data.attributes.types.AnnotationsAttr;
import jadx.core.dex.nodes.RootNode;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import ru.blackfan.bfscan.config.ProcessorConfig;
import ru.blackfan.bfscan.helpers.Helpers;
import ru.blackfan.bfscan.parsing.httprequests.Constants;

public class FieldAnnotationProcessor {

    private final RootNode rootNode;
    private String fieldName;
    private String defaultValue;

    public FieldAnnotationProcessor(RootNode rootNode) {
        this.rootNode = rootNode;
    }

    public void process(AnnotationsAttr aList, String initialFieldName, String initialDefaultValue) {
        this.fieldName = initialFieldName;
        this.defaultValue = initialDefaultValue;

        for (IAnnotation a : aList.getAll()) {
            processAnnotation(a);
        }
    }

    private void processAnnotation(IAnnotation annotation) {
        Map<String, EncodedValue> annotationValues = annotation.getValues();
        String annotationClass = AnnotationUtils.getAnnotationClass(annotation, rootNode);
        if(Constants.Ignore.IGNORE_ANNOTATIONS.contains(annotationClass)) {
            return;
        }
        switch (annotationClass) {
            case Constants.Serialization.GSON_SERIALIZED_NAME, Constants.Serialization.KOTLINX_SERIALNAME, 
                    Constants.JaxRs.PARAM_FORM, Constants.Jakarta.PARAM_FORM, 
                    Constants.JaxRs.PARAM_QUERY, Constants.Jakarta.PARAM_QUERY, 
                    Constants.JaxRs.PARAM_COOKIE, Constants.Jakarta.PARAM_COOKIE, 
                    Constants.JaxRs.PARAM_HEADER, Constants.Jakarta.PARAM_HEADER, 
                    Constants.JaxRs.PARAM_MATRIX, Constants.Jakarta.PARAM_MATRIX, 
                    Constants.Serialization.JAX_JSONBPROPERTY, 
                    Constants.Serialization.JAKARTA_JSONBPROPERTY,
                    Constants.Serialization.APACHE_JOHNZON_PROPERTY,
                    Constants.Spring.BIND_PARAM, Constants.Serialization.CODEHAUS_JSON_PROPERTY,
                    Constants.Serialization.XSTREAM_ALIAS -> {
                processNameValue(annotationValues);
            }
            case Constants.Serialization.SQUAREUP_MOSHI_JSON, Constants.Serialization.FASTJSON_JSONFIELD,
                    Constants.Serialization.FASTJSON2_JSONFIELD, Constants.Serialization.JAX_COLUMN-> {
                processNameName(annotationValues);
            }
            case Constants.Serialization.JAKARTA_XML_ELEMENT, Constants.Serialization.JAX_XML_ELEMENT -> {
                processXmlElement(annotationValues);
            }
            case Constants.Serialization.JACKSON_JSON_PROPERTY -> {
                processJsonProperty(annotationValues);
            }
            case Constants.Swagger.API_PARAM, Constants.Swagger.API_MODEL_PROPERTY -> {
                processSwaggerAnnotation(annotationValues);
            }
            case Constants.OpenApi.SCHEMA -> {
                processOpenApiAnnotation(annotationValues);
            }
            default -> {
                if(ProcessorConfig.getInstance().isMinifiedAnnotationsSupport()) {
                    minifiedAnnotation(annotationValues);
                }
            }
        }
    }

    private void minifiedAnnotation(Map<String, EncodedValue> values) {
        EncodedValue value = AnnotationUtils.getValue(values, List.of("value", "name", "a"));
        if (value != null && value.getType() == EncodedType.ENCODED_STRING) {
            String potentialFieldName = Helpers.stringWrapper(value);
            if (potentialFieldName.matches("^[a-zA-Z_][a-zA-Z0-9_]*$")) {
                fieldName = potentialFieldName;
            }
        }
    }

    private void processNameValue(Map<String, EncodedValue> values) {
        EncodedValue value = AnnotationUtils.getValue(values, List.of("value", "a"));
        if (value != null) {
            fieldName = Helpers.stringWrapper(value);
        }
    }

    private void processNameName(Map<String, EncodedValue> values) {
        EncodedValue value = AnnotationUtils.getValue(values, List.of("name"));
        if (value != null) {
            fieldName = Helpers.stringWrapper(value);
        }
    }

    private void processXmlElement(Map<String, EncodedValue> values) {
        EncodedValue value = AnnotationUtils.getValue(values, List.of("name"));
        if (value != null) {
            fieldName = Helpers.stringWrapper(value);
        }
        value = AnnotationUtils.getValue(values, List.of("defaultValue"));
        if (value != null) {
            defaultValue = Helpers.stringWrapper(value);
        }
    }

    private void processJsonProperty(Map<String, EncodedValue> values) {
        EncodedValue value = AnnotationUtils.getValue(values, List.of("value"));
        if (value != null) {
            fieldName = Helpers.stringWrapper(value);
        }
        value = AnnotationUtils.getValue(values, List.of("defaultValue"));
        if (value != null) {
            defaultValue = Helpers.stringWrapper(value);
        }
    }

    private void processSwaggerAnnotation(Map<String, EncodedValue> values) {
        EncodedValue value = AnnotationUtils.getValue(values, List.of("name"));
        if (value != null) {
            fieldName = Helpers.stringWrapper(value);
        }
        value = AnnotationUtils.getValue(values, List.of("defaultValue", "example", "allowableValues"));
        if (value != null) {
            defaultValue = Helpers.stringWrapper(value);
        }
    }

    private void processOpenApiAnnotation(Map<String, EncodedValue> values) {
        EncodedValue value = AnnotationUtils.getValue(values, List.of("name"));
        if (value != null) {
            fieldName = Helpers.stringWrapper(value);
        }
        value = AnnotationUtils.getValue(values, List.of("defaultValue", "example"));
        if (value != null) {
            defaultValue = Helpers.stringWrapper(value);
        } else {
            value = AnnotationUtils.getValue(values, List.of("allowableValues", "examples"));
            if (value != null) {
                ArrayList<EncodedValue> examples = (ArrayList) value.getValue();
                if (!examples.isEmpty()) {
                    defaultValue = Helpers.stringWrapper(examples.get(0));
                }
            }
        }
    }

    public String getFieldName() {
        return fieldName;
    }

    public String getDefaultValue() {
        return defaultValue;
    }
}
