package ru.blackfan.bfscan.parsing.httprequests;

import jadx.api.JadxDecompiler;
import jadx.api.ResourceFile;
import jadx.api.ResourcesLoader;
import jadx.core.dex.nodes.ClassNode;
import jadx.core.utils.exceptions.JadxException;
import jadx.core.xmlgen.ResContainer;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.io.IOException;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.XMLConstants;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.InputSource;
import ru.blackfan.bfscan.helpers.Helpers;
import ru.blackfan.bfscan.parsing.httprequests.processors.AnnotationUtils;

public class ResourceProcessor {

    private static final Logger logger = LoggerFactory.getLogger(ResourceProcessor.class);

    private final JadxDecompiler jadx;
    private final DocumentBuilderFactory factory;
    private final String apiBasePath;
    private final String apiHost;

    public ResourceProcessor(JadxDecompiler jadx, URI apiUrl) {
        this.jadx = jadx;
        this.apiBasePath = apiUrl.getPath();
        this.apiHost = apiUrl.getHost();
        this.factory = DocumentBuilderFactory.newInstance();
        this.factory.setValidating(false);
        try {
            this.factory.setFeature(XMLConstants.FEATURE_SECURE_PROCESSING, true);
            this.factory.setFeature("http://apache.org/xml/features/nonvalidating/load-external-dtd", false);
            this.factory.setFeature("http://xml.org/sax/features/external-general-entities", false);
            this.factory.setFeature("http://xml.org/sax/features/external-parameter-entities", false);
        } catch (ParserConfigurationException ex) {
            logger.error("ParserConfigurationException", ex);
        }
    }

    public List<MultiHTTPRequest> processResources() {
        List<MultiHTTPRequest> multiRequests = new ArrayList();
        for (ResourceFile resFile : jadx.getResources()) {
            ResContainer resContainer = resFile.loadContent();
            if (resContainer.getDataType() == ResContainer.DataType.RES_LINK) {
                try {
                    ResourcesLoader.decodeStream(resContainer.getResLink(), (size, is) -> {
                        multiRequests.addAll(processFile(resFile.getDeobfName(), is));
                        return null;
                    });
                } catch (JadxException ex) {
                    logger.error("Error processing file " + resFile.getDeobfName(), ex);
                }
            }
            if (resContainer.getDataType() == ResContainer.DataType.TEXT) {
                multiRequests.addAll(processFile(resFile.getDeobfName(), new ByteArrayInputStream(resContainer.getText().getCodeStr().getBytes(StandardCharsets.UTF_8))));
            }
        }
        return multiRequests;
    }

    public List<MultiHTTPRequest> processFile(String name, InputStream is) {
        List<MultiHTTPRequest> multiRequests = new ArrayList<>();
        switch (Helpers.getFileExtension(name)) {
            case "xml" -> {
                multiRequests.addAll(processXml(name, is));
            }
            case "zip", "jar" -> {
                try {
                    ZipFile zip = Helpers.inputSteamToZipFile(is);
                    List<ZipEntry> entries = (List<ZipEntry>) Collections.list(zip.entries());

                    List<MultiHTTPRequest> results = entries.parallelStream()
                            .filter(entry -> !entry.isDirectory())
                            .map(entry -> {
                                try (InputStream zipEntryIs = zip.getInputStream(entry)) {
                                    return processFile(name + "#" + entry.getName(), zipEntryIs);
                                } catch (IOException ex) {
                                    logger.error("Error processing zip entry " + entry.getName() + " in " + name, ex);
                                    return Collections.<MultiHTTPRequest>emptyList();
                                }
                            })
                            .flatMap(List::stream)
                            .collect(Collectors.toList());

                    multiRequests.addAll(results);

                    try {
                        zip.close();
                    } catch (IOException ex) {
                        logger.warn("Error closing zip file " + name, ex);
                    }
                } catch (IOException ex) {
                    logger.error("Error processing file " + name, ex);
                }
            }
        }
        return multiRequests;
    }

    private List<MultiHTTPRequest> processXml(String name, InputStream is) {
        try {
            DocumentBuilder builder = factory.newDocumentBuilder();
            InputSource inputSource = new InputSource(is);
            Document document = builder.parse(inputSource);
            Element root = document.getDocumentElement();

            String rootTagName = root.getTagName();

            if (null != rootTagName) {
                switch (rootTagName) {
                    case "web-app", "web-fragment" -> {
                        return processWebXml(name, document);
                    }
                    case "struts-config" -> {
                        return processStrutsConfigXml(name, document);
                    }
                    case "struts" -> {
                        return processStrutsXml(name, document);
                    }
                    case "Configure" -> {
                        return processJettyXml(name, document);
                    }
                    default -> {
                    }
                }
            }
        } catch (Exception ex) {
            logger.error("Error parsing requests from " + name, ex);
        }
        return new ArrayList();
    }

    private List<MultiHTTPRequest> processWebXml(String name, Document document) {
        List<MultiHTTPRequest> multiRequests = new ArrayList();
        try {
            Map<String, String> servletClasses = new HashMap<>();

            NodeList servlets = document.getElementsByTagName("servlet");
            for (int i = 0; i < servlets.getLength(); i++) {
                Element servlet = (Element) servlets.item(i);
                String servletName = getElementText(servlet, "servlet-name");
                String servletClassOrJsp = getElementText(servlet, "servlet-class");
                if (servletClassOrJsp.isEmpty()) {
                    servletClassOrJsp = getElementText(servlet, "jsp-file");
                }
                if (!servletName.isEmpty() && !servletClassOrJsp.isEmpty()) {
                    servletClasses.put(servletName, servletClassOrJsp);
                }
            }

            NodeList mappings = document.getElementsByTagName("servlet-mapping");
            for (int i = 0; i < mappings.getLength(); i++) {
                Element mapping = (Element) mappings.item(i);
                String servletName = getElementText(mapping, "servlet-name");
                NodeList urlPatterns = mapping.getElementsByTagName("url-pattern");
                for (int j = 0; j < urlPatterns.getLength(); j++) {
                    String urlPattern = urlPatterns.item(j).getTextContent();
                    String servletClass = servletClasses.getOrDefault(servletName, "UNKNOWN");

                    MultiHTTPRequest webXmlRequest = new MultiHTTPRequest(apiHost, apiBasePath, servletClass, name);
                    webXmlRequest.addAdditionalInformation("web.xml Servlet");
                    webXmlRequest.setPath(urlPattern, false);
                    multiRequests.add(webXmlRequest);
                }
            }
        } catch (Exception ex) {
            logger.error("Error parsing requests from web.xml", ex);
        }
        return multiRequests;
    }

    private List<MultiHTTPRequest> processStrutsConfigXml(String name, Document document) {
        List<MultiHTTPRequest> multiRequests = new ArrayList();
        try {
            Map<String, String> forms = new HashMap();
            Map<String, Map<String, Object>> dynaFormsParameters = new HashMap();
            NodeList formBeansList = document.getElementsByTagName("form-beans");
            if (formBeansList.getLength() > 0) {
                Element formBeansNode = (Element) formBeansList.item(0);
                NodeList formBeans = formBeansNode.getElementsByTagName("form-bean");
                for (int i = 0; i < formBeans.getLength(); i++) {
                    Element formBean = (Element) formBeans.item(i);
                    String formName = formBean.getAttribute("name");
                    String formType = formBean.getAttribute("type");
                    if (!formName.isEmpty() && !formType.isEmpty() && !formType.equals(Constants.Struts.VALID_FORM)) {
                        forms.put(formName, formType);
                        if (Constants.Struts.DYNA_FORM.contains(formType)) {
                            Map<String, Object> formsParameters = new HashMap();
                            NodeList formProperties = formBean.getElementsByTagName("form-property");
                            for (int j = 0; j < formProperties.getLength(); j++) {
                                Element formProperty = (Element) formProperties.item(j);
                                String propertyName = formProperty.getAttribute("name");
                                String propertyType = formProperty.getAttribute("type");
                                if (!propertyName.isEmpty() && !propertyType.isEmpty()) {
                                    if (propertyType.endsWith("[]")) {
                                        propertyType = propertyType.substring(0, propertyType.length() - 2);
                                    }
                                    formsParameters.put(propertyName, AnnotationUtils.classNameToDefaultValue(propertyName, propertyType, jadx.getRoot(), new HashSet<>(), false));
                                }
                            }
                            dynaFormsParameters.put(formName, formsParameters);
                        }
                    }
                }
            }

            NodeList mappings = document.getElementsByTagName("action-mappings");
            if (mappings.getLength() > 0) {
                Element mappingsNode = (Element) mappings.item(0);
                NodeList actions = mappingsNode.getElementsByTagName("action");
                for (int i = 0; i < actions.getLength(); i++) {
                    Element action = (Element) actions.item(i);
                    String path = action.getAttribute("path");
                    String type = action.getAttribute("type");
                    String actionName = action.getAttribute("name");
                    if (type.isEmpty()) {
                        type = "unknown-class";
                    }
                    if (!path.isEmpty()) {
                        MultiHTTPRequest strutsXmlRequest = new MultiHTTPRequest(apiHost, apiBasePath, type, name);
                        strutsXmlRequest.addAdditionalInformation("Struts Config Action");
                        strutsXmlRequest.setPath(path + ".action", false);
                        if (forms.containsKey(actionName)) {
                            final String formClass = forms.get(actionName);
                            Map<String, Object> parameters = null;
                            if (!Constants.Struts.DYNA_FORM.contains(formClass)) {
                                if (!formClass.isEmpty()) {
                                    ClassNode classNode = Helpers.loadClass(jadx.getRoot(), formClass);
                                    if (classNode != null) {
                                        parameters = AnnotationUtils.classToRequestParameters(classNode, false, jadx.getRoot());
                                    }
                                }
                            } else {
                                if (dynaFormsParameters.containsKey(actionName)) {
                                    Map<String, Object> formParameters = dynaFormsParameters.get(actionName);
                                    if (!formParameters.isEmpty()) {
                                        parameters = formParameters;
                                    }
                                }
                            }
                            AnnotationUtils.appendParametersToRequest(strutsXmlRequest, parameters);
                        }
                        multiRequests.add(strutsXmlRequest);
                    }
                }
            }
        } catch (Exception ex) {
            logger.error("Error parsing requests from struts-config.xml", ex);
        }
        return multiRequests;
    }

    private List<MultiHTTPRequest> processStrutsXml(String name, Document document) {
        List<MultiHTTPRequest> multiRequests = new ArrayList();
        try {
            NodeList packages = document.getElementsByTagName("package");
            for (int i = 0; i < packages.getLength(); i++) {
                Element pkg = (Element) packages.item(0);
                NodeList actions = pkg.getElementsByTagName("action");
                for (int j = 0; j < actions.getLength(); j++) {
                    Element action = (Element) actions.item(j);
                    String actionName = action.getAttribute("name");
                    String actionClass = action.getAttribute("class");
                    if (!actionName.isEmpty()) {
                        Map<String, Object> parameters = new HashMap();
                        if (!actionClass.isEmpty()) {
                            ClassNode classNode = Helpers.loadClass(jadx.getRoot(), actionClass);
                            if (classNode != null) {
                                parameters = AnnotationUtils.classToRequestParameters(classNode, false, jadx.getRoot());
                            }
                        } else {
                            actionClass = "unknown-class";
                        }

                        MultiHTTPRequest strutsXmlRequest = new MultiHTTPRequest(apiHost, apiBasePath, actionClass, name);
                        strutsXmlRequest.addAdditionalInformation("Struts Action");
                        strutsXmlRequest.setPath(actionName + ".action", false);
                        AnnotationUtils.appendParametersToRequest(strutsXmlRequest, parameters);
                        multiRequests.add(strutsXmlRequest);
                    }
                }
            }
        } catch (Exception ex) {
            logger.error("Error parsing requests from struts.xml", ex);
        }
        return multiRequests;
    }

    /* Support only the simplest syntax for adding servlets */
    private List<MultiHTTPRequest> processJettyXml(String name, Document document) {
        List<MultiHTTPRequest> multiRequests = new ArrayList<>();

        try {
            NodeList calls = document.getElementsByTagName("Call");
            for (int i = 0; i < calls.getLength(); i++) {
                Element call = (Element) calls.item(i);
                if ("addServlet".equals(call.getAttribute("name")) || "addServletWithMapping".equals(call.getAttribute("name"))) {
                    NodeList args = call.getElementsByTagName("Arg");
                    if (args.getLength() >= 2) {
                        if (!hasElementChildren((Element) args.item(0)) && !hasElementChildren((Element) args.item(1))) {
                            String servletName = args.item(0).getTextContent();
                            String urlPattern = args.item(1).getTextContent();

                            MultiHTTPRequest jettyRequest = new MultiHTTPRequest(apiHost, apiBasePath, servletName, name);
                            jettyRequest.addAdditionalInformation("jetty.xml Servlet");
                            jettyRequest.setPath(urlPattern, false);
                            multiRequests.add(jettyRequest);
                        }
                    }
                }
            }
        } catch (Exception ex) {
            logger.error("Error parsing requests from jetty.xml", ex);
        }
        return multiRequests;
    }

    private boolean hasElementChildren(Element element) {
        NodeList children = element.getChildNodes();
        for (int i = 0; i < children.getLength(); i++) {
            if (children.item(i).getNodeType() == Node.ELEMENT_NODE) {
                return true;
            }
        }
        return false;
    }

    private static String getElementText(Element element, String tagName) {
        NodeList nodeList = element.getElementsByTagName(tagName);
        if (nodeList.getLength() > 0) {
            return nodeList.item(0).getTextContent().trim();
        }
        return "";
    }
}
