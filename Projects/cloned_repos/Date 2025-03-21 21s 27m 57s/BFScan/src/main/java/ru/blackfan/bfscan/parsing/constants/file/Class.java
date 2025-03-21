package ru.blackfan.bfscan.parsing.constants.file;

import jadx.api.JavaClass;
import jadx.api.JavaField;
import jadx.api.JavaMethod;
import jadx.api.plugins.input.data.annotations.EncodedType;
import jadx.api.plugins.input.data.annotations.EncodedValue;
import jadx.api.plugins.input.data.annotations.IAnnotation;
import jadx.api.plugins.input.data.attributes.JadxAttrType;
import jadx.api.plugins.input.data.attributes.types.AnnotationsAttr;
import jadx.core.dex.instructions.ConstStringNode;
import jadx.core.dex.nodes.InsnNode;
import jadx.core.dex.nodes.MethodNode;
import jadx.core.utils.exceptions.DecodeException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ru.blackfan.bfscan.config.ConfigLoader;
import ru.blackfan.bfscan.helpers.Helpers;
import ru.blackfan.bfscan.helpers.KeyValuePair;
import ru.blackfan.bfscan.parsing.httprequests.Constants;

public class Class {

    private static final Logger logger = LoggerFactory.getLogger(Class.class);
    public static final List<String> EXCLUDED_PACKAGES = ConfigLoader.getExcludedPackages();

    public static Set<KeyValuePair> process(String fileName, JavaClass cls) {
        Set<KeyValuePair> keyValuePairs = new HashSet<>();
        logger.debug("Process class " + fileName);

        for (String pkg : EXCLUDED_PACKAGES) {
            if (fileName.startsWith(pkg)) {
                return keyValuePairs;
            }
        }
        for (JavaField field : cls.getFields()) {
            Set<String> stringConstants = new HashSet<>();
            EncodedValue constVal = field.getFieldNode().get(JadxAttrType.CONSTANT_VALUE);
            if (constVal != null) {
                parseStrings(constVal, stringConstants, cls.getClassNode().getAlias());
                for (String str : stringConstants) {
                    logger.debug("    Class field constant: " + str);
                    keyValuePairs.add(new KeyValuePair(field.getName(), str));
                }
            }
        }

        AnnotationsAttr classAList = cls.getClassNode().get(JadxAttrType.ANNOTATION_LIST);
        if (classAList != null && !classAList.isEmpty()) {
            Set<String> stringConstants = new HashSet<>();
            extractStringFromAnnotations(classAList, stringConstants, cls.getClassNode().getAlias());
            for (String str : stringConstants) {
                logger.debug("    Class annotation constant: " + str);
                keyValuePairs.add(new KeyValuePair("class_annotation", str));
            }
        }

        for (JavaMethod mth : cls.getMethods()) {
            logger.debug("  Method: " + mth.getName());
            if (!mth.getName().equals("toString")) {
                Set<String> stringConstants = getStringConstants(mth);
                for (String str : stringConstants) {
                    logger.debug("    String constant: " + str);
                    keyValuePairs.add(new KeyValuePair(mth.getName(), str));
                }
            }
        }
        return keyValuePairs;
    }

    private static Set<String> getStringConstants(JavaMethod method) {
        Set<String> stringConstants = new HashSet<>();
        try {
            MethodNode methodNode = method.getMethodNode();
            if (methodNode != null) {
                methodNode.load();
                if (methodNode.getInstructions() == null && methodNode.getInsnsCount() != 0) {
                    methodNode.reload();
                }
                InsnNode[] instructions = methodNode.getInstructions();
                if (instructions != null) {
                    for (InsnNode insn : instructions) {
                        if (insn instanceof ConstStringNode constStrInsn) {
                            addConstant(constStrInsn.getString(), stringConstants, methodNode.getAlias());
                        }
                    }
                } else {
                    logger.debug("No instructions found for method: {}", method.getName());
                }

                AnnotationsAttr aList = methodNode.get(JadxAttrType.ANNOTATION_LIST);
                if (aList != null && !aList.isEmpty()) {
                    extractStringFromAnnotations(aList, stringConstants, methodNode.getAlias());
                }
            }
            return stringConstants;
        } catch (DecodeException ex) {
            logger.error("Method DecodeException", ex);
        }
        return stringConstants;
    }

    private static void extractStringFromAnnotations(AnnotationsAttr annotations, Set<String> stringConstants, String methodName) {
        for (IAnnotation annotation : annotations.getAll()) {
            if (!annotation.getAnnotationClass().startsWith(Constants.KOTLIN_METADATA)) {
                for (Map.Entry<String, EncodedValue> entry : annotation.getValues().entrySet()) {
                    parseStrings(entry.getValue(), stringConstants, methodName);
                }
            }
        }
    }

    private static void parseStrings(EncodedValue value, Set<String> stringConstants, String methodName) {
        if (value != null) {
            if (value.getType() == EncodedType.ENCODED_STRING) {
                addConstant(Helpers.stringWrapper(value), stringConstants, methodName);
            } else if (value.getType() == EncodedType.ENCODED_ARRAY) {
                for (Object obj : (ArrayList) value.getValue()) {
                    if (((EncodedValue) obj).getType() == EncodedType.ENCODED_STRING) {
                        addConstant(Helpers.stringWrapper((EncodedValue) obj), stringConstants, methodName);
                    }
                }
            }
        }
    }

    private static void addConstant(String s, Set<String> stringConstants, String methodName) {
        if (!Helpers.isJVMSig(s) && !s.equals(methodName)) {
            stringConstants.add(s);
        }
    }
}
