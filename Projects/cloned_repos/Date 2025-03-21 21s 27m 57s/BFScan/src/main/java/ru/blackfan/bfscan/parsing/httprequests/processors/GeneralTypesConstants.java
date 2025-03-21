package ru.blackfan.bfscan.parsing.httprequests.processors;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class GeneralTypesConstants {

    public static final String JAVA_BOOLEAN = "java.lang.Boolean";
    public static final String JAVA_MAP = "java.util.Map";
    public static final String JAVA_OPTIONAL = "java.util.Optional";
    public static final String JAVA_STRING = "java.lang.String";

    public static final Set<String> MULTIPART_CLASSES = new HashSet<>(Arrays.asList(
            "org.springframework.web.multipart.MultipartFile",
            "javax.servlet.http.Part",
            "jakarta.servlet.http.Part",
            "io.micronaut.http.multipart.MultipartFile"
    ));

    public static final Set<String> JSON_OBJECT_CLASSES = new HashSet<>(Arrays.asList(
            "org.json.JSONObject",
            "com.google.gson.JsonObject",
            "com.fasterxml.jackson.databind.JsonNode",
            "com.fasterxml.jackson.databind.ObjectNode"
    ));

    public static final Set<String> JSON_ARRAY_CLASSES = new HashSet<>(Arrays.asList(
            "org.json.JSONArray",
            "com.google.gson.JsonArray",
            "com.fasterxml.jackson.databind.ArrayNode"
    ));

    public static final Set<String> NUMBER_CLASSES = new HashSet<>(Arrays.asList(
            "java.lang.Byte",
            "java.lang.Short",
            "java.lang.Integer",
            "java.lang.Long",
            "java.lang.Float",
            "java.lang.Double",
            "java.math.BigInteger",
            "java.math.BigDecimal"
    ));

    public static final Set<String> ARRAY_CLASSES = new HashSet<>(Arrays.asList(
            "java.util.List",
            "java.util.Set",
            "java.util.Collection",
            "java.util.ArrayList",
            "java.util.HashSet",
            "java.util.LinkedList",
            "java.util.TreeSet",
            "java.util.EnumSet"
    ));

    public static final Set<String> DATE_CLASSES = new HashSet<>(Arrays.asList(
            "java.util.Date",
            "java.time.LocalDate",
            "java.time.ZonedDateTime"
    ));
}
