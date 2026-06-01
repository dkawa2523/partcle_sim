import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;

import java.io.BufferedWriter;
import java.io.PrintWriter;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class ComsolParticleCaseExporter {
    private static Path mphPath;
    private static Path configPath;
    private static Path outDir;
    private static String caseName = "comsol_particle_case";
    private static String mode = "all";
    private static boolean exportMesh = true;
    private static boolean exportFields = true;
    private static boolean exportGridFieldSamples = true;
    private static boolean exportMeshFieldSamples = false;
    private static String meshFieldSamplesFilename = "mesh_field_samples.csv";
    private static String dataset = "dset1";
    private static String meshTag = "mesh1";
    private static int spatialDim = 2;
    private static int solnum = -1;
    private static int[] solnums = new int[0];
    private static double[] timeValues = new double[0];
    private static boolean exportDataTable = false;
    private static boolean exportDataTableRequired = false;
    private static String dataExportDataset = "";
    private static String dataExportFilename = "comsol_data_export.csv";
    private static String dataExportInnerInput = "";
    private static int[] dataExportSolnums = new int[0];
    private static double[] dataExportTimeValues = new double[0];
    private static List<String> dataExportExpressions = new ArrayList<>();
    private static boolean enableParticleStatusData = false;
    private static boolean enableWallExtraSteps = false;
    private static boolean enableParticleReleaseStatistics = false;
    private static boolean requireRuntimeOptionApplication = false;
    private static boolean runStudy = false;
    private static String studyTag = "std1";
    private static String wallAccuracyOrder = "";
    private static String maxBounce = "";
    private static String coordinateModelUnit = "m";
    private static double coordinateScaleMPerModelUnit = 1.0;
    private static String[] axisNames = new String[]{"x", "y"};
    private static double[] axisMin = new double[]{0.0, 0.0, 0.0};
    private static double[] axisMax = new double[]{1.0, 1.0, 1.0};
    private static int[] axisCount = new int[]{51, 51, 51};
    private static List<String> required = Arrays.asList("ux", "uy", "mu", "E_x", "E_y");
    private static Map<String, List<String>> expressions = new LinkedHashMap<>();

    public static void main(String[] args) throws Exception {
        loadOptions(args);
        loadConfig(configPath);
        Files.createDirectories(outDir);

        Model model = ModelUtil.load("particle_export_model", mphPath.toString());
        try {
            List<String[]> runtimeOptions = applyRuntimeOptions(model);
            String[] studyRun = runConfiguredStudy(model);
            writeRuntimeOptionsReport(outDir.resolve("runtime_options_report.json"), runtimeOptions, studyRun);
            writeInventories(model);
            if (exportFields && ("all".equals(mode) || "fields".equals(mode))) {
                Map<String, String> selected = selectExpressions(model, probeCoordinates());
                writeExpressionInventory(outDir.resolve("expression_inventory.json"), selected);
                if (exportGridFieldSamples) {
                    writeFieldSamples(outDir.resolve("field_samples.csv"), model, selected);
                }
                if (exportMeshFieldSamples) {
                    writeMeshFieldSamples(outDir.resolve(meshFieldSamplesFilename), model, selected);
                }
                writeManifest(outDir.resolve("export_manifest.json"), selected);
            } else {
                writeExpressionInventory(outDir.resolve("expression_inventory.json"), new LinkedHashMap<String, String>());
                writeManifest(outDir.resolve("export_manifest.json"), new LinkedHashMap<String, String>());
            }
            if (exportDataTable) {
                writeConfiguredDataExport(model);
            }
        } finally {
            try {
                ModelUtil.disconnect();
            } catch (Throwable ignored) {
            }
        }
    }

    private static void loadOptions(String[] args) {
        Map<String, String> map = new LinkedHashMap<>();
        if (args != null) {
            for (int i = 0; i + 1 < args.length; i += 2) {
                map.put(args[i].replaceFirst("^-+", ""), args[i + 1]);
            }
        }
        mphPath = Paths.get(firstNonEmpty(map.get("mph"), System.getenv("COMSOL_PARTICLE_MPH"), "model.mph"));
        configPath = Paths.get(firstNonEmpty(map.get("config"), System.getenv("COMSOL_PARTICLE_CONFIG"), "external/comsol_particle_export/config/export_case.example.json"));
        outDir = Paths.get(firstNonEmpty(map.get("outdir"), System.getenv("COMSOL_PARTICLE_OUTDIR"), "_external_exports/comsol_particle_case"));
    }

    private static void loadConfig(Path path) throws Exception {
        String text = new String(Files.readAllBytes(path), StandardCharsets.UTF_8);
        caseName = jsonString(text, "case_name", caseName);
        mode = jsonString(text, "mode", mode);
        exportMesh = jsonBoolean(text, "export_mesh", exportMesh);
        exportFields = jsonBoolean(text, "export_fields", exportFields);
        exportGridFieldSamples = jsonBoolean(text, "export_grid_field_samples", exportGridFieldSamples);
        exportMeshFieldSamples = jsonBoolean(text, "export_mesh_field_samples", exportMeshFieldSamples);
        meshFieldSamplesFilename = jsonString(text, "mesh_field_samples_filename", meshFieldSamplesFilename);
        dataset = jsonString(text, "dataset", dataset);
        meshTag = jsonString(text, "mesh_tag", meshTag);
        spatialDim = (int) jsonDouble(text, "spatial_dim", spatialDim);
        solnum = (int) jsonDouble(text, "solnum", solnum);
        solnums = jsonIntArrayConfig(text, "solnums", new int[0]);
        timeValues = jsonDoubleArray(text, "time_values", jsonDoubleArray(text, "times", new double[0]));
        exportDataTable = jsonBoolean(text, "export_data_table", exportDataTable);
        exportDataTableRequired = jsonBoolean(text, "export_data_table_required", exportDataTableRequired);
        dataExportDataset = jsonString(text, "data_export_dataset", dataExportDataset);
        dataExportFilename = jsonString(text, "data_export_filename", dataExportFilename);
        dataExportInnerInput = jsonString(text, "data_export_innerinput", dataExportInnerInput);
        dataExportSolnums = jsonIntArrayConfig(text, "data_export_solnums", new int[0]);
        dataExportTimeValues = jsonDoubleArray(text, "data_export_time_values", jsonDoubleArray(text, "data_export_times", new double[0]));
        enableParticleStatusData = jsonBoolean(text, "enable_particle_status_data", enableParticleStatusData);
        enableWallExtraSteps = jsonBoolean(text, "enable_wall_extra_steps", enableWallExtraSteps);
        enableParticleReleaseStatistics = jsonBoolean(text, "enable_particle_release_statistics", enableParticleReleaseStatistics);
        requireRuntimeOptionApplication = jsonBoolean(text, "require_runtime_option_application", requireRuntimeOptionApplication);
        runStudy = jsonBoolean(text, "run_study", runStudy);
        studyTag = jsonString(text, "study_tag", studyTag);
        wallAccuracyOrder = jsonString(text, "wall_accuracy_order", wallAccuracyOrder);
        maxBounce = jsonString(text, "max_wall_interactions_per_time_step", maxBounce);
        coordinateModelUnit = jsonString(text, "coordinate_model_unit", coordinateModelUnit);
        coordinateScaleMPerModelUnit = jsonDouble(text, "coordinate_scale_m_per_model_unit", coordinateScaleMPerModelUnit);
        if (spatialDim < 1 || spatialDim > 3) {
            throw new IllegalArgumentException("spatial_dim must be 1, 2, or 3");
        }
        axisNames = jsonStringArray(text, "axis_names", defaultAxisNames(spatialDim));
        if (axisNames.length != spatialDim) {
            throw new IllegalArgumentException("axis_names length must match spatial_dim");
        }
        for (int d = 0; d < spatialDim; d++) {
            axisMin[d] = jsonDouble(text, "axis_" + d + "_min", axisMin[d]);
            axisMax[d] = jsonDouble(text, "axis_" + d + "_max", axisMax[d]);
            axisCount[d] = (int) jsonDouble(text, "axis_" + d + "_count", axisCount[d]);
            if (axisCount[d] < 2) {
                throw new IllegalArgumentException("axis_" + d + "_count must be at least 2");
            }
        }
        expressions = parseExpressionLists(text);
        expressions.remove("axis_names");
        dataExportExpressions = firstExpressionList(
            expressions,
            "data_export_expr",
            "data_export_expressions",
            "particle_result_expressions"
        );
        expressions.remove("data_export_expr");
        expressions.remove("data_export_expressions");
        expressions.remove("particle_result_expressions");
        expressions.remove("data_export_descriptions");
        expressions.remove("data_export_units");
        if (exportDataTable && dataExportExpressions.isEmpty()) {
            dataExportExpressions.add("1");
        }
        if (expressions.containsKey("required")) {
            required = new ArrayList<>(expressions.get("required"));
            expressions.remove("required");
        }
    }

    private static void writeInventories(Model model) throws Exception {
        writeModelInventory(model, outDir.resolve("model_inventory.json"));
        writeMaterialInventory(model, outDir.resolve("material_inventory.json"));
        writeSelectionInventory(model, outDir.resolve("selection_inventory.json"));
        writeStudyInventory(model, outDir.resolve("study_inventory.json"));
        writeDatasetInventory(model, outDir.resolve("dataset_inventory.json"));
        writePhysicsFeatureInventory(model, outDir.resolve("physics_feature_inventory.json"));
        writeParticleReleaseInventory(model, outDir.resolve("particle_release_inventory.json"));
        writeMethodList(outDir.resolve("model_methods.txt"), model);
        if (exportMesh && ("all".equals(mode) || "inventory".equals(mode) || "fields".equals(mode))) {
            exportMesh(model, outDir.resolve("mesh.mphtxt"));
        }
    }

    private static boolean runtimeOptionsRequested() {
        return enableParticleStatusData
            || enableWallExtraSteps
            || enableParticleReleaseStatistics
            || !wallAccuracyOrder.trim().isEmpty()
            || !maxBounce.trim().isEmpty();
    }

    private static List<String[]> applyRuntimeOptions(Model model) {
        List<String[]> rows = new ArrayList<>();
        if (!runtimeOptionsRequested()) {
            return rows;
        }
        String[] components = listTags(safeCall(model, "component"));
        for (String component : components) {
            Object comp = safeCall(model, "component", component);
            String[] physicsTags = listTags(safeCall(comp, "physics"));
            for (String physicsTag : physicsTags) {
                Object physics = safeCall(comp, "physics", physicsTag);
                String physicsLabel = stringOrEmpty(safeCall(physics, "label"));
                String physicsType = firstNonEmpty(
                    stringOrEmpty(safeCall(physics, "getType")),
                    stringOrEmpty(safeCall(physics, "type"))
                );
                if (!isParticleTracingPhysics(physicsTag, physicsLabel, physicsType)) {
                    continue;
                }
                if (enableParticleStatusData) {
                    rows.add(setRuntimeOption(physics, component, physicsTag, physicsLabel, "StoreParticleStatusData", "1"));
                }
                if (enableWallExtraSteps) {
                    rows.add(setRuntimeOption(physics, component, physicsTag, physicsLabel, "StoreExtra", "1"));
                }
                if (enableParticleReleaseStatistics) {
                    rows.add(setRuntimeOption(physics, component, physicsTag, physicsLabel, "StoreParticleReleaseStatistics", "1"));
                }
                if (!wallAccuracyOrder.trim().isEmpty()) {
                    rows.add(setRuntimeOption(physics, component, physicsTag, physicsLabel, "WallAccuracyOrder", wallAccuracyOrder.trim()));
                }
                if (!maxBounce.trim().isEmpty()) {
                    rows.add(setRuntimeOption(physics, component, physicsTag, physicsLabel, "MaxBounce", maxBounce.trim()));
                }
            }
        }
        if (requireRuntimeOptionApplication) {
            if (rows.isEmpty()) {
                throw new RuntimeException("No particle-tracing physics interface was found for required runtime options");
            }
            for (String[] row : rows) {
                if (!Boolean.parseBoolean(row[6])) {
                    throw new RuntimeException(
                        "Required COMSOL runtime option could not be set: "
                            + row[0] + "/" + row[1] + "/" + row[3] + "=" + row[4]
                    );
                }
            }
        }
        return rows;
    }

    private static String[] setRuntimeOption(
        Object physics,
        String component,
        String physicsTag,
        String physicsLabel,
        String propertyName,
        String value
    ) {
        boolean success = setPhysicsPropertyIfPossible(physics, propertyName, value);
        String actualValue = physicsPropertySetting(physics, propertyName);
        return new String[]{
            component,
            physicsTag,
            physicsLabel,
            propertyName,
            value,
            actualValue,
            String.valueOf(success)
        };
    }

    private static String[] runConfiguredStudy(Model model) {
        if (!runStudy) {
            return new String[]{"false", studyTag, "false", "0", ""};
        }
        long start = System.nanoTime();
        try {
            Object study = call(model, "study", studyTag);
            call(study, "run");
            long durationMs = (System.nanoTime() - start) / 1000000L;
            return new String[]{"true", studyTag, "true", String.valueOf(durationMs), ""};
        } catch (Throwable t) {
            throw new RuntimeException("COMSOL study run failed for " + studyTag + ": " + t.toString(), t);
        }
    }

    private static void writeRuntimeOptionsReport(Path out, List<String[]> options, String[] studyRun) throws Exception {
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            w.write("{\n");
            w.write("  \"source_kind\": \"external_comsol_particle_export_runtime_options\",\n");
            w.write("  \"case_name\": " + json(caseName) + ",\n");
            w.write("  \"runtime_options_requested\": " + runtimeOptionsRequested() + ",\n");
            w.write("  \"require_runtime_option_application\": " + requireRuntimeOptionApplication + ",\n");
            w.write("  \"particle_physics_option_applications\": [\n");
            for (int i = 0; i < options.size(); i++) {
                if (i > 0) {
                    w.write(",\n");
                }
                String[] row = options.get(i);
                w.write("    {\n");
                w.write("      \"component_tag\": " + json(row[0]) + ",\n");
                w.write("      \"physics_tag\": " + json(row[1]) + ",\n");
                w.write("      \"physics_label\": " + json(row[2]) + ",\n");
                w.write("      \"property\": " + json(row[3]) + ",\n");
                w.write("      \"value\": " + json(row[4]) + ",\n");
                w.write("      \"success\": " + Boolean.parseBoolean(row[6]) + ",\n");
                w.write("      \"actual_value\": " + json(row[5]) + "\n");
                w.write("    }");
            }
            w.write("\n  ],\n");
            w.write("  \"study_run\": {\n");
            w.write("    \"requested\": " + Boolean.parseBoolean(studyRun[0]) + ",\n");
            w.write("    \"study_tag\": " + json(studyRun[1]) + ",\n");
            w.write("    \"success\": " + Boolean.parseBoolean(studyRun[2]) + ",\n");
            w.write("    \"duration_ms\": " + studyRun[3] + ",\n");
            w.write("    \"error\": " + json(studyRun[4]) + "\n");
            w.write("  }\n");
            w.write("}\n");
        }
    }

    private static void writeModelInventory(Model model, Path out) throws Exception {
        String[] components = listTags(safeCall(model, "component"));
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            w.write("{\n");
            w.write("  \"source_kind\": \"external_comsol_particle_export_inventory\",\n");
            w.write("  \"case_name\": " + json(caseName) + ",\n");
            w.write("  \"mph_path\": " + json(mphPath.toString()) + ",\n");
            w.write("  \"mph_sha256\": " + json(sha256(mphPath)) + ",\n");
            w.write("  \"comsol_version\": " + json(stringOrEmpty(safeCall(model, "getComsolVersion"))) + ",\n");
            w.write("  \"title\": " + json(stringOrEmpty(safeCall(model, "title"))) + ",\n");
            w.write("  \"parameter_names\": " + jsonArray(listTagsLike(safeCall(model, "param"), "varnames")) + ",\n");
            w.write("  \"component_tags\": " + jsonArray(components) + ",\n");
            w.write("  \"study_tags\": " + jsonArray(listTags(safeCall(model, "study"))) + ",\n");
            w.write("  \"solver_tags\": " + jsonArray(listTags(safeCall(model, "sol"))) + ",\n");
            w.write("  \"dataset_tags\": " + jsonArray(listTags(safeCall(safeCall(model, "result"), "dataset"))) + ",\n");
            w.write("  \"components\": [\n");
            for (int i = 0; i < components.length; i++) {
                Object comp = safeCall(model, "component", components[i]);
                if (i > 0) {
                    w.write(",\n");
                }
                w.write("    {\n");
                w.write("      \"tag\": " + json(components[i]) + ",\n");
                w.write("      \"physics_tags\": " + jsonArray(listTags(safeCall(comp, "physics"))) + ",\n");
                w.write("      \"mesh_tags\": " + jsonArray(listTags(safeCall(comp, "mesh"))) + ",\n");
                w.write("      \"geometry_tags\": " + jsonArray(listTags(safeCall(comp, "geom"))) + ",\n");
                w.write("      \"selection_tags\": " + jsonArray(listTags(safeCall(comp, "selection"))) + "\n");
                w.write("    }");
            }
            w.write("\n  ]\n");
            w.write("}\n");
        }
    }

    private static void writeMaterialInventory(Model model, Path out) throws Exception {
        Object materials = safeCall(model, "material");
        String[] tags = listTags(materials);
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            w.write("{\n");
            w.write("  \"source_kind\": \"external_comsol_particle_export_material_inventory\",\n");
            w.write("  \"materials\": [\n");
            for (int i = 0; i < tags.length; i++) {
                Object material = safeCall(model, "material", tags[i]);
                Object selection = safeCall(material, "selection");
                if (i > 0) {
                    w.write(",\n");
                }
                w.write("    {\n");
                w.write("      \"tag\": " + json(tags[i]) + ",\n");
                w.write("      \"label\": " + json(stringOrEmpty(safeCall(material, "label"))) + ",\n");
                w.write("      \"name\": " + json(stringOrEmpty(safeCall(material, "name"))) + ",\n");
                w.write("      \"selection_entities\": " + jsonIntArray(selectionEntities(selection)) + "\n");
                w.write("    }");
            }
            w.write("\n  ]\n");
            w.write("}\n");
        }
    }

    private static void writeSelectionInventory(Model model, Path out) throws Exception {
        List<String> rows = new ArrayList<>();
        collectSelections(rows, model, "", "model");
        String[] components = listTags(safeCall(model, "component"));
        for (String component : components) {
            collectSelections(rows, safeCall(model, "component", component), component, "component");
        }
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            w.write("{\n");
            w.write("  \"source_kind\": \"external_comsol_particle_export_selection_inventory\",\n");
            w.write("  \"selections\": [\n");
            for (int i = 0; i < rows.size(); i++) {
                if (i > 0) {
                    w.write(",\n");
                }
                w.write(rows.get(i));
            }
            w.write("\n  ]\n");
            w.write("}\n");
        }
    }

    private static void collectSelections(List<String> rows, Object owner, String componentTag, String ownerKind) {
        String[] tags = listTags(safeCall(owner, "selection"));
        for (String tag : tags) {
            Object selection = safeCall(owner, "selection", tag);
            StringBuilder sb = new StringBuilder();
            sb.append("    {\n");
            sb.append("      \"owner_kind\": ").append(json(ownerKind)).append(",\n");
            sb.append("      \"component_tag\": ").append(json(componentTag)).append(",\n");
            sb.append("      \"tag\": ").append(json(tag)).append(",\n");
            sb.append("      \"label\": ").append(json(stringOrEmpty(safeCall(selection, "label")))).append(",\n");
            sb.append("      \"name\": ").append(json(stringOrEmpty(safeCall(selection, "name")))).append(",\n");
            sb.append("      \"entities\": ").append(jsonIntArray(selectionEntities(selection))).append("\n");
            sb.append("    }");
            rows.add(sb.toString());
        }
    }

    private static void writeStudyInventory(Model model, Path out) throws Exception {
        String[] studyTags = listTags(safeCall(model, "study"));
        List<String> rows = new ArrayList<>();
        for (String studyTag : studyTags) {
            Object study = safeCall(model, "study", studyTag);
            String[] featureTags = listTags(safeCall(study, "feature"));
            StringBuilder sb = new StringBuilder();
            sb.append("    {\n");
            sb.append("      \"tag\": ").append(json(studyTag)).append(",\n");
            sb.append("      \"label\": ").append(json(stringOrEmpty(safeCall(study, "label")))).append(",\n");
            sb.append("      \"type\": ").append(json(firstNonEmpty(stringOrEmpty(safeCall(study, "getType")), stringOrEmpty(safeCall(study, "type"))))).append(",\n");
            sb.append("      \"property_names\": ").append(jsonArray(featurePropertyNames(study))).append(",\n");
            sb.append("      \"property_values\": ").append(featurePropertyValuesJson(study)).append(",\n");
            sb.append("      \"features\": [\n");
            for (int i = 0; i < featureTags.length; i++) {
                Object feature = safeCall(study, "feature", featureTags[i]);
                if (i > 0) {
                    sb.append(",\n");
                }
                sb.append("        {\n");
                sb.append("          \"tag\": ").append(json(featureTags[i])).append(",\n");
                sb.append("          \"label\": ").append(json(stringOrEmpty(safeCall(feature, "label")))).append(",\n");
                sb.append("          \"type\": ").append(json(firstNonEmpty(stringOrEmpty(safeCall(feature, "getType")), stringOrEmpty(safeCall(feature, "type"))))).append(",\n");
                sb.append("          \"property_names\": ").append(jsonArray(featurePropertyNames(feature))).append(",\n");
                sb.append("          \"property_values\": ").append(featurePropertyValuesJson(feature)).append("\n");
                sb.append("        }");
            }
            sb.append("\n      ]\n");
            sb.append("    }");
            rows.add(sb.toString());
        }
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            w.write("{\n");
            w.write("  \"source_kind\": \"external_comsol_particle_export_study_inventory\",\n");
            w.write("  \"studies\": [\n");
            for (int i = 0; i < rows.size(); i++) {
                if (i > 0) {
                    w.write(",\n");
                }
                w.write(rows.get(i));
            }
            w.write("\n  ]\n");
            w.write("}\n");
        }
    }

    private static void writeDatasetInventory(Model model, Path out) throws Exception {
        Object datasets = safeCall(safeCall(model, "result"), "dataset");
        String[] datasetTags = listTags(datasets);
        List<String> rows = new ArrayList<>();
        for (String datasetTag : datasetTags) {
            Object datasetObj = safeCall(safeCall(safeCall(model, "result"), "dataset"), datasetTag);
            StringBuilder sb = new StringBuilder();
            sb.append("    {\n");
            sb.append("      \"tag\": ").append(json(datasetTag)).append(",\n");
            sb.append("      \"label\": ").append(json(stringOrEmpty(safeCall(datasetObj, "label")))).append(",\n");
            sb.append("      \"type\": ").append(json(firstNonEmpty(stringOrEmpty(safeCall(datasetObj, "getType")), stringOrEmpty(safeCall(datasetObj, "type"))))).append(",\n");
            sb.append("      \"property_names\": ").append(jsonArray(featurePropertyNames(datasetObj))).append(",\n");
            sb.append("      \"property_values\": ").append(featurePropertyValuesJson(datasetObj)).append("\n");
            sb.append("    }");
            rows.add(sb.toString());
        }
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            w.write("{\n");
            w.write("  \"source_kind\": \"external_comsol_particle_export_dataset_inventory\",\n");
            w.write("  \"datasets\": [\n");
            for (int i = 0; i < rows.size(); i++) {
                if (i > 0) {
                    w.write(",\n");
                }
                w.write(rows.get(i));
            }
            w.write("\n  ]\n");
            w.write("}\n");
        }
    }

    private static void writePhysicsFeatureInventory(Model model, Path out) throws Exception {
        String[] components = listTags(safeCall(model, "component"));
        List<String> interfaceRows = new ArrayList<>();
        List<String> rows = new ArrayList<>();
        for (String component : components) {
            Object comp = safeCall(model, "component", component);
            String[] physicsTags = listTags(safeCall(comp, "physics"));
            for (String physicsTag : physicsTags) {
                Object physics = safeCall(comp, "physics", physicsTag);
                String physicsLabel = stringOrEmpty(safeCall(physics, "label"));
                String physicsType = firstNonEmpty(
                    stringOrEmpty(safeCall(physics, "getType")),
                    stringOrEmpty(safeCall(physics, "type"))
                );
                StringBuilder physicsSb = new StringBuilder();
                physicsSb.append("    {\n");
                physicsSb.append("      \"component_tag\": ").append(json(component)).append(",\n");
                physicsSb.append("      \"physics_tag\": ").append(json(physicsTag)).append(",\n");
                physicsSb.append("      \"physics_label\": ").append(json(physicsLabel)).append(",\n");
                physicsSb.append("      \"physics_type\": ").append(json(physicsType)).append(",\n");
                physicsSb.append("      \"property_names\": ").append(jsonArray(featurePropertyNames(physics))).append(",\n");
                physicsSb.append("      \"property_values\": ").append(featurePropertyValuesJson(physics)).append("\n");
                physicsSb.append("    }");
                interfaceRows.add(physicsSb.toString());
                String[] featureTags = listTags(safeCall(physics, "feature"));
                for (String featureTag : featureTags) {
                    Object feature = safeCall(physics, "feature", featureTag);
                    String label = stringOrEmpty(safeCall(feature, "label"));
                    String type = firstNonEmpty(
                        stringOrEmpty(safeCall(feature, "getType")),
                        stringOrEmpty(safeCall(feature, "type"))
                    );
                    String forceKind = classifyForceKind(physicsTag, physicsLabel, physicsType, featureTag, label, type);
                    StringBuilder sb = new StringBuilder();
                    sb.append("    {\n");
                    sb.append("      \"component_tag\": ").append(json(component)).append(",\n");
                    sb.append("      \"physics_tag\": ").append(json(physicsTag)).append(",\n");
                    sb.append("      \"physics_label\": ").append(json(physicsLabel)).append(",\n");
                    sb.append("      \"physics_type\": ").append(json(physicsType)).append(",\n");
                    sb.append("      \"feature_tag\": ").append(json(featureTag)).append(",\n");
                    sb.append("      \"label\": ").append(json(label)).append(",\n");
                    sb.append("      \"type\": ").append(json(type)).append(",\n");
                    sb.append("      \"force_kind\": ").append(json(forceKind)).append(",\n");
                    sb.append("      \"selection_entities\": ").append(jsonIntArray(selectionEntities(safeCall(feature, "selection")))).append(",\n");
                    sb.append("      \"property_names\": ").append(jsonArray(featurePropertyNames(feature))).append(",\n");
                    sb.append("      \"known_settings\": ").append(featureKnownSettingsJson(feature)).append(",\n");
                    sb.append("      \"property_values\": ").append(featurePropertyValuesJson(feature)).append("\n");
                    sb.append("    }");
                    rows.add(sb.toString());
                }
            }
        }
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            w.write("{\n");
            w.write("  \"source_kind\": \"external_comsol_particle_export_physics_feature_inventory\",\n");
            w.write("  \"physics_interfaces\": [\n");
            for (int i = 0; i < interfaceRows.size(); i++) {
                if (i > 0) {
                    w.write(",\n");
                }
                w.write(interfaceRows.get(i));
            }
            w.write("\n  ],\n");
            w.write("  \"features\": [\n");
            for (int i = 0; i < rows.size(); i++) {
                if (i > 0) {
                    w.write(",\n");
                }
                w.write(rows.get(i));
            }
            w.write("\n  ]\n");
            w.write("}\n");
        }
    }

    private static void writeParticleReleaseInventory(Model model, Path out) throws Exception {
        String[] components = listTags(safeCall(model, "component"));
        List<String> rows = new ArrayList<>();
        for (String component : components) {
            Object comp = safeCall(model, "component", component);
            String[] physicsTags = listTags(safeCall(comp, "physics"));
            for (String physicsTag : physicsTags) {
                Object physics = safeCall(comp, "physics", physicsTag);
                String physicsLabel = stringOrEmpty(safeCall(physics, "label"));
                String physicsType = firstNonEmpty(
                    stringOrEmpty(safeCall(physics, "getType")),
                    stringOrEmpty(safeCall(physics, "type"))
                );
                if (!isParticleTracingPhysics(physicsTag, physicsLabel, physicsType)) {
                    continue;
                }
                String[] featureTags = listTags(safeCall(physics, "feature"));
                for (String featureTag : featureTags) {
                    Object feature = safeCall(physics, "feature", featureTag);
                    String label = stringOrEmpty(safeCall(feature, "label"));
                    String type = firstNonEmpty(
                        stringOrEmpty(safeCall(feature, "getType")),
                        stringOrEmpty(safeCall(feature, "type"))
                    );
                    String kind = classifyParticleReleaseKind(physicsTag, physicsLabel, physicsType, featureTag, label, type);
                    if ("other".equals(kind)) {
                        continue;
                    }
                    StringBuilder sb = new StringBuilder();
                    sb.append("    {\n");
                    sb.append("      \"component_tag\": ").append(json(component)).append(",\n");
                    sb.append("      \"physics_tag\": ").append(json(physicsTag)).append(",\n");
                    sb.append("      \"physics_label\": ").append(json(physicsLabel)).append(",\n");
                    sb.append("      \"physics_type\": ").append(json(physicsType)).append(",\n");
                    sb.append("      \"feature_tag\": ").append(json(featureTag)).append(",\n");
                    sb.append("      \"label\": ").append(json(label)).append(",\n");
                    sb.append("      \"type\": ").append(json(type)).append(",\n");
                    sb.append("      \"release_kind\": ").append(json(kind)).append(",\n");
                    sb.append("      \"selection_entities\": ").append(jsonIntArray(selectionEntities(safeCall(feature, "selection")))).append(",\n");
                    sb.append("      \"property_names\": ").append(jsonArray(featurePropertyNames(feature))).append(",\n");
                    sb.append("      \"known_settings\": ").append(particleReleaseSettingsJson(feature)).append(",\n");
                    sb.append("      \"property_values\": ").append(featurePropertyValuesJson(feature)).append("\n");
                    sb.append("    }");
                    rows.add(sb.toString());
                }
            }
        }
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            w.write("{\n");
            w.write("  \"source_kind\": \"external_comsol_particle_export_release_inventory\",\n");
            w.write("  \"features\": [\n");
            for (int i = 0; i < rows.size(); i++) {
                if (i > 0) {
                    w.write(",\n");
                }
                w.write(rows.get(i));
            }
            w.write("\n  ]\n");
            w.write("}\n");
        }
    }

    private static boolean isParticleTracingPhysics(String... values) {
        String text = String.join(" ", values).toLowerCase(Locale.ROOT);
        return containsAny(text, "particle", "tracing", "fpt");
    }

    private static String classifyParticleReleaseKind(String... values) {
        String text = String.join(" ", values).toLowerCase(Locale.ROOT);
        if (containsAny(text, "grid") && containsAny(text, "release", "inlet", "inject", "source")) {
            return "release_grid";
        }
        if (containsAny(text, "release", "inlet", "inject", "source", "initial position", "initial coordinates")) {
            return "release";
        }
        if (containsAny(text, "initial velocity", "velocity direction", "velocity magnitude")) {
            return "initial_velocity";
        }
        if (containsAny(text, "particle properties", "particle property", "diameter", "mass", "density")) {
            return "particle_properties";
        }
        return "other";
    }

    private static String classifyForceKind(String... values) {
        String text = String.join(" ", values).toLowerCase(Locale.ROOT);
        if (containsAny(text, "thermophor", "thermophoretic")) {
            return "thermophoresis";
        }
        if (containsAny(text, "dielectrophor", "dielectrophoretic", "dep")) {
            return "dielectrophoresis";
        }
        if (containsAny(text, "saffman", "lift")) {
            return "lift";
        }
        if (containsAny(text, "gravity", "gravit")) {
            return "gravity";
        }
        if (containsAny(text, "brownian", "langevin")) {
            return "brownian";
        }
        if (containsAny(text, "drag", "stokes", "epstein")) {
            return "drag";
        }
        if (containsAny(text, "electric", "electrostatic", "coulomb")) {
            return "electric";
        }
        if (containsAny(text, "magnetic", "lorentz")) {
            return "magnetic";
        }
        return "other";
    }

    private static boolean containsAny(String text, String... needles) {
        for (String needle : needles) {
            if (text.contains(needle)) {
                return true;
            }
        }
        return false;
    }

    private static String featureKnownSettingsJson(Object feature) {
        String[] keys = new String[]{
            "F", "Fx", "Fy", "Fz",
            "g", "g_const", "gvec",
            "T", "rho", "mu", "eta",
            "k", "kg", "kp", "Cs", "Cm", "Ct",
            "E", "Ex", "Ey", "Ez", "V",
            "epsilonr", "epsilonrp", "sigma", "sigmap", "freq",
            "u", "v", "w", "U", "walllift"
        };
        List<String> items = new ArrayList<>();
        for (String key : keys) {
            String value = featureSetting(feature, key);
            if (value == null || value.trim().isEmpty()) {
                continue;
            }
            items.add(json(key) + ": " + json(value));
        }
        return "{" + String.join(", ", items) + "}";
    }

    private static String particleReleaseSettingsJson(Object feature) {
        String[] keys = new String[]{
            "N", "n", "Np", "nump", "number", "nParticles", "npart",
            "t", "t0", "t1", "tlist", "times", "rt", "release_times", "releaseTime", "trelease", "tRelease",
            "period", "frequency", "f", "phase", "pulse", "duration", "tstart", "tend", "dt",
            "grid", "gridtype", "gridType", "Nx", "Ny", "Nz", "Nr", "Nz_grid", "n0", "n1", "n2",
            "x0", "y0", "z0", "r0", "x", "y", "z", "r", "coord", "coords",
            "vx0", "vy0", "vz0", "vr0", "v0", "v", "speed", "direction", "normal",
            "diameter", "dp", "radius", "rp", "rho", "density", "mass", "mp",
            "charge", "q", "material", "selection", "distrib", "distribution"
        };
        List<String> items = new ArrayList<>();
        for (String key : keys) {
            String value = featureSetting(feature, key);
            if (value == null || value.trim().isEmpty()) {
                continue;
            }
            items.add(json(key) + ": " + json(value));
        }
        return "{" + String.join(", ", items) + "}";
    }

    private static String featurePropertyValuesJson(Object feature) {
        List<String> items = new ArrayList<>();
        for (String key : featurePropertyNames(feature)) {
            String value = featureSetting(feature, key);
            if (value == null || value.trim().isEmpty()) {
                continue;
            }
            items.add(json(key) + ": " + json(value));
        }
        return "{" + String.join(", ", items) + "}";
    }

    private static String[] featurePropertyNames(Object feature) {
        for (String method : new String[]{"properties", "getProperties", "propertyNames", "getPropertyNames"}) {
            Object value = safeCall(feature, method);
            if (value instanceof String[]) {
                return (String[]) value;
            }
        }
        return new String[0];
    }

    private static String featureSetting(Object feature, String key) {
        for (String method : new String[]{"getString", "get"}) {
            try {
                Object value = call(feature, method, key);
                if (value != null) {
                    return String.valueOf(value);
                }
            } catch (Throwable ignored) {
            }
        }
        return "";
    }

    private static void exportMesh(Model model, Path out) {
        List<String> errors = new ArrayList<>();
        try {
            writeMphtxtFromMeshSequence(call(model, "mesh", meshTag), out);
            return;
        } catch (Throwable t) {
            errors.add(t.toString());
        }
        for (String component : listTags(safeCall(model, "component"))) {
            try {
                Object comp = call(model, "component", component);
                writeMphtxtFromMeshSequence(call(comp, "mesh", meshTag), out);
                return;
            } catch (Throwable t) {
                errors.add(t.toString());
            }
        }
        throw new RuntimeException("Could not export mesh.mphtxt. Errors: " + errors);
    }

    private static double[][] meshVertices(Model model) {
        List<String> errors = new ArrayList<>();
        try {
            return normalizeVertices((double[][]) call(call(model, "mesh", meshTag), "getVertex"));
        } catch (Throwable t) {
            errors.add(t.toString());
        }
        for (String component : listTags(safeCall(model, "component"))) {
            try {
                Object comp = call(model, "component", component);
                return normalizeVertices((double[][]) call(call(comp, "mesh", meshTag), "getVertex"));
            } catch (Throwable t) {
                errors.add(t.toString());
            }
        }
        throw new RuntimeException("Could not read mesh vertices. Errors: " + errors);
    }

    private static void writeMphtxtFromMeshSequence(Object mesh, Path out) throws Exception {
        double[][] vertices = normalizeVertices((double[][]) call(mesh, "getVertex"));
        String[] types = (String[]) call(mesh, "getTypes");
        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(out, StandardCharsets.UTF_8))) {
            int sdim = vertices.length == 0 ? 0 : vertices[0].length;
            writer.println(sdim + " # sdim");
            writer.println(vertices.length + " # number of mesh vertices");
            writer.println("# Mesh vertex coordinates");
            for (double[] vertex : vertices) {
                for (int d = 0; d < sdim; d++) {
                    if (d > 0) {
                        writer.print(" ");
                    }
                    writer.print(String.format(Locale.US, "%.17g", vertex[d]));
                }
                writer.println();
            }
            writer.println(types.length + " # number of element types");
            for (int typeIndex = 0; typeIndex < types.length; typeIndex++) {
                String type = types[typeIndex];
                int[] entity = (int[]) call(mesh, "getElemEntity", type);
                int[][] elems = normalizeElements((int[][]) call(mesh, "getElem", type), entity.length);
                int nvp = elems.length == 0 ? 0 : elems[0].length;
                writer.println(typeIndex + " " + type + " # type name");
                writer.println(nvp + " # number of vertices per element");
                writer.println(elems.length + " # number of elements");
                writer.println("# Elements");
                for (int[] elem : elems) {
                    for (int j = 0; j < nvp; j++) {
                        if (j > 0) {
                            writer.print(" ");
                        }
                        writer.print(elem[j]);
                    }
                    writer.println();
                }
                writer.println(entity.length + " # number of geometric entity indices");
                writer.println("# Geometric entity indices");
                for (int value : entity) {
                    writer.println(value);
                }
            }
        }
    }

    private static Map<String, String> selectExpressions(Model model, double[][] probeCoords) {
        Map<String, String> selected = new LinkedHashMap<>();
        Map<String, String> failures = new LinkedHashMap<>();
        for (Map.Entry<String, List<String>> entry : expressions.entrySet()) {
            String key = entry.getKey();
            List<String> fail = new ArrayList<>();
            for (String expr : entry.getValue()) {
                String tag = "inv_" + sanitize(key);
                try {
                    Object interp = createInterp(model, tag, expr);
                    double value = evalFirstFinite(interp, probeCoords);
                    if (Double.isFinite(value)) {
                        selected.put(key, expr);
                        failures.put(key, "");
                        break;
                    }
                    fail.add(expr + ": non-finite");
                } catch (Throwable t) {
                    fail.add(expr + ": " + t.getClass().getSimpleName() + ": " + t.getMessage());
                } finally {
                    removeNumerical(model, tag);
                }
            }
            if (!selected.containsKey(key)) {
                failures.put(key, String.join("; ", fail));
            }
        }
        for (String key : required) {
            if (!selected.containsKey(key)) {
                throw new RuntimeException("Required COMSOL expression not found for " + key + ": " + failures.get(key));
            }
        }
        return selected;
    }

    private static Object createInterp(Model model, String tag, String expr) {
        int activeSolnum = solnums.length > 0 ? solnums[0] : solnum;
        double activeTime = timeValues.length > 0 ? timeValues[0] : Double.NaN;
        return createInterp(model, tag, expr, activeSolnum, activeTime);
    }

    private static Object createInterp(Model model, String tag, String expr, int activeSolnum, double activeTime) {
        Object result = call(model, "result");
        Object numerical = call(result, "numerical");
        try {
            call(numerical, "remove", tag);
        } catch (Throwable ignored) {
        }
        call(numerical, "create", tag, "Interp");
        Object interp = call(result, "numerical", tag);
        call(interp, "set", "data", dataset);
        if (activeSolnum > 0) {
            call(interp, "set", "solnum", new int[]{activeSolnum});
        }
        if (Double.isFinite(activeTime)) {
            boolean applied = false;
            for (String key : new String[]{"t", "time"}) {
                try {
                    call(interp, "set", key, new double[]{activeTime});
                    applied = true;
                    break;
                } catch (Throwable ignored) {
                }
            }
            if (!applied) {
                try {
                    call(interp, "set", "t", String.format(Locale.US, "%.17g", activeTime));
                } catch (Throwable ignored) {
                }
            }
        }
        call(interp, "set", "expr", new String[]{expr});
        return interp;
    }

    private static void removeNumerical(Model model, String tag) {
        try {
            call(call(call(model, "result"), "numerical"), "remove", tag);
        } catch (Throwable ignored) {
        }
    }

    private static void writeFieldSamples(Path out, Model model, Map<String, String> selected) throws Exception {
        double[][] coords = gridCoordinates();
        int contextCount = sampleContextCount();
        boolean includeContextColumns = shouldWriteSampleContextColumns();
        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(out, StandardCharsets.UTF_8))) {
            if (includeContextColumns) {
                writer.print("time_s,solnum,");
            }
            for (int d = 0; d < spatialDim; d++) {
                if (d > 0) {
                    writer.print(",");
                }
                writer.print(axisNames[d]);
            }
            writer.print(",valid_mask");
            for (String key : selected.keySet()) {
                writer.print(",");
                writer.print(key);
            }
            writer.println();
            for (int context = 0; context < contextCount; context++) {
                int activeSolnum = sampleSolnum(context);
                double activeTime = sampleTime(context);
                Map<String, Object> features = new LinkedHashMap<>();
                for (String key : selected.keySet()) {
                    features.put(
                        key,
                        createInterp(
                            model,
                            "grid_" + sanitize(key) + "_" + context,
                            selected.get(key),
                            activeSolnum,
                            activeTime
                        )
                    );
                }
                Map<String, double[]> valuesByKey = new LinkedHashMap<>();
                for (String key : selected.keySet()) {
                    valuesByKey.put(key, evalMany(features.get(key), coords));
                }
                for (int i = 0; i < coords[0].length; i++) {
                    boolean valid = true;
                    for (String key : selected.keySet()) {
                        if (required.contains(key) && !Double.isFinite(valuesByKey.get(key)[i])) {
                            valid = false;
                        }
                    }
                    if (includeContextColumns) {
                        writer.print(Double.isFinite(activeTime) ? String.format(Locale.US, "%.17g", activeTime) : "");
                        writer.print(",");
                        writer.print(activeSolnum > 0 ? String.valueOf(activeSolnum) : "");
                        writer.print(",");
                    }
                    for (int d = 0; d < spatialDim; d++) {
                        if (d > 0) {
                            writer.print(",");
                        }
                        writer.print(String.format(Locale.US, "%.17g", coords[d][i]));
                    }
                    writer.print(",");
                    writer.print(valid ? "1" : "0");
                    for (String key : selected.keySet()) {
                        double value = valuesByKey.get(key)[i];
                        writer.print(",");
                        writer.print(Double.isFinite(value) ? String.format(Locale.US, "%.17g", value) : "NaN");
                    }
                    writer.println();
                }
                for (String key : selected.keySet()) {
                    removeNumerical(model, "grid_" + sanitize(key) + "_" + context);
                }
            }
        } finally {
            for (String key : selected.keySet()) {
                removeNumerical(model, "grid_" + sanitize(key));
            }
        }
    }

    private static void writeMeshFieldSamples(Path out, Model model, Map<String, String> selected) throws Exception {
        if (spatialDim != 2) {
            throw new RuntimeException("mesh_field_samples export currently supports spatial_dim=2 only");
        }
        double[][] vertices = meshVertices(model);
        if (vertices.length == 0 || vertices[0].length < spatialDim) {
            throw new RuntimeException("mesh vertices do not expose the configured spatial_dim");
        }
        double[][] coords = new double[spatialDim][vertices.length];
        for (int i = 0; i < vertices.length; i++) {
            for (int d = 0; d < spatialDim; d++) {
                coords[d][i] = vertices[i][d];
            }
        }

        int contextCount = sampleContextCount();
        boolean includeContextColumns = shouldWriteSampleContextColumns();
        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(out, StandardCharsets.UTF_8))) {
            if (includeContextColumns) {
                writer.print("time_s,solnum,");
            }
            writer.print("vertex_id");
            for (int d = 0; d < spatialDim; d++) {
                writer.print(",");
                writer.print(axisNames[d]);
            }
            writer.print(",valid_mask");
            for (String key : selected.keySet()) {
                writer.print(",");
                writer.print(key);
            }
            writer.println();

            for (int context = 0; context < contextCount; context++) {
                int activeSolnum = sampleSolnum(context);
                double activeTime = sampleTime(context);
                Map<String, Object> features = new LinkedHashMap<>();
                for (String key : selected.keySet()) {
                    features.put(
                        key,
                        createInterp(
                            model,
                            "mesh_" + sanitize(key) + "_" + context,
                            selected.get(key),
                            activeSolnum,
                            activeTime
                        )
                    );
                }
                Map<String, double[]> valuesByKey = new LinkedHashMap<>();
                for (String key : selected.keySet()) {
                    valuesByKey.put(key, evalMany(features.get(key), coords));
                }
                for (int i = 0; i < vertices.length; i++) {
                    boolean valid = true;
                    for (String key : selected.keySet()) {
                        if (required.contains(key) && !Double.isFinite(valuesByKey.get(key)[i])) {
                            valid = false;
                        }
                    }
                    if (includeContextColumns) {
                        writer.print(Double.isFinite(activeTime) ? String.format(Locale.US, "%.17g", activeTime) : "");
                        writer.print(",");
                        writer.print(activeSolnum > 0 ? String.valueOf(activeSolnum) : "");
                        writer.print(",");
                    }
                    writer.print(i);
                    for (int d = 0; d < spatialDim; d++) {
                        writer.print(",");
                        writer.print(String.format(Locale.US, "%.17g", coords[d][i]));
                    }
                    writer.print(",");
                    writer.print(valid ? "1" : "0");
                    for (String key : selected.keySet()) {
                        double value = valuesByKey.get(key)[i];
                        writer.print(",");
                        writer.print(Double.isFinite(value) ? String.format(Locale.US, "%.17g", value) : "NaN");
                    }
                    writer.println();
                }
                for (String key : selected.keySet()) {
                    removeNumerical(model, "mesh_" + sanitize(key) + "_" + context);
                }
            }
        } finally {
            for (String key : selected.keySet()) {
                removeNumerical(model, "mesh_" + sanitize(key));
            }
        }
    }

    private static void writeExpressionInventory(Path out, Map<String, String> selected) throws Exception {
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            w.write("{\n");
            w.write("  \"source_kind\": \"external_comsol_particle_export_expression_inventory\",\n");
            w.write("  \"dataset\": " + json(dataset) + ",\n");
            w.write("  \"required\": " + jsonArray(required.toArray(new String[0])) + ",\n");
            w.write("  \"selected\": {\n");
            int i = 0;
            for (String key : expressions.keySet()) {
                if (i++ > 0) {
                    w.write(",\n");
                }
                w.write("    " + json(key) + ": {");
                w.write("\"expression\": " + json(selected.containsKey(key) ? selected.get(key) : "") + ", ");
                w.write("\"dataset\": " + json(selected.containsKey(key) ? dataset : "") + ", ");
                w.write("\"available\": " + selected.containsKey(key));
                w.write("}");
            }
            w.write("\n  }\n");
            w.write("}\n");
        }
    }

    private static void writeManifest(Path out, Map<String, String> selected) throws Exception {
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            w.write("{\n");
            w.write("  \"source_kind\": \"external_comsol_particle_export\",\n");
            w.write("  \"case_name\": " + json(caseName) + ",\n");
            w.write("  \"mph_path\": " + json(mphPath.toString()) + ",\n");
            w.write("  \"mph_sha256\": " + json(sha256(mphPath)) + ",\n");
            w.write("  \"dataset\": " + json(dataset) + ",\n");
            w.write("  \"mesh_tag\": " + json(meshTag) + ",\n");
            w.write("  \"spatial_dim\": " + spatialDim + ",\n");
            w.write("  \"axis_names\": " + jsonArray(axisNames) + ",\n");
            w.write("  \"coordinate_model_unit\": " + json(coordinateModelUnit) + ",\n");
            w.write("  \"coordinate_scale_m_per_model_unit\": " + jsonNumber(coordinateScaleMPerModelUnit) + ",\n");
            w.write("  \"grid_shape\": " + jsonIntArray(Arrays.copyOf(axisCount, spatialDim)) + ",\n");
            w.write("  \"export_grid_field_samples\": " + exportGridFieldSamples + ",\n");
            w.write("  \"export_mesh_field_samples\": " + exportMeshFieldSamples + ",\n");
            w.write("  \"mesh_field_samples_filename\": " + json(meshFieldSamplesFilename) + ",\n");
            w.write("  \"solnum\": " + solnum + ",\n");
            w.write("  \"solnums\": " + jsonIntArray(activeSolnumsForManifest()) + ",\n");
            w.write("  \"time_values\": " + jsonNumberArray(timeValues) + ",\n");
            w.write("  \"field_sample_context_count\": " + sampleContextCount() + ",\n");
            w.write("  \"expression_mapping\": {\n");
            int i = 0;
            for (Map.Entry<String, String> entry : selected.entrySet()) {
                if (i++ > 0) {
                    w.write(",\n");
                }
                w.write("    " + json(entry.getKey()) + ": " + json(entry.getValue()));
            }
            w.write("\n  }\n");
            w.write("}\n");
        }
    }

    private static void writeConfiguredDataExport(Model model) throws Exception {
        String targetDataset = firstNonEmpty(dataExportDataset, dataset);
        Path target = outDir.resolve(dataExportFilename).toAbsolutePath();
        List<String> errors = new ArrayList<>();
        boolean success = false;
        String exportTag = "codex_data_export";
        try {
            Files.deleteIfExists(target);
        } catch (Throwable t) {
            errors.add("delete existing output: " + t.toString());
        }
        try {
            Object result = call(model, "result");
            Object exports = call(result, "export");
            removeExport(exports, exportTag);
            createDataExport(exports, exportTag, targetDataset);
            Object feature = call(result, "export", exportTag);
            setIfPossible(feature, "data", targetDataset);
            setIfPossible(feature, "filename", target.toString());
            setIfPossible(feature, "expr", dataExportExpressions.toArray(new String[0]));
            setIfPossible(feature, "fullprec", "on");
            setIfPossible(feature, "header", "on");
            setIfPossible(feature, "includecoords", true);
            setIfPossible(feature, "includecoords", "on");
            setIfPossible(feature, "includenan", true);
            setIfPossible(feature, "includenan", "on");
            setIfPossible(feature, "location", "fromdataset");
            setIfPossible(feature, "struct", "spreadsheet");
            setIfPossible(feature, "separator", ",");
            setIfPossible(feature, "exporttype", "text");
            applyDataExportSolutionSelection(feature);
            call(feature, "run");
            success = Files.exists(target);
            if (!success) {
                errors.add("COMSOL Data export finished but did not create " + target);
            }
            removeExport(exports, exportTag);
        } catch (Throwable t) {
            errors.add(t.toString());
        } finally {
            writeDataExportReport(
                outDir.resolve("data_export_report.json"),
                targetDataset,
                target,
                success,
                errors
            );
        }
        if (!success && exportDataTableRequired) {
            throw new RuntimeException("Required COMSOL Data export failed: " + String.join("; ", errors));
        }
    }

    private static void createDataExport(Object exports, String tag, String targetDataset) {
        List<String> errors = new ArrayList<>();
        try {
            call(exports, "create", tag, targetDataset, "Data");
            return;
        } catch (Throwable t) {
            errors.add("create(tag,dataset,Data): " + t.toString());
        }
        try {
            call(exports, "create", tag, "Data");
            return;
        } catch (Throwable t) {
            errors.add("create(tag,Data): " + t.toString());
        }
        throw new RuntimeException("Could not create COMSOL Data export feature: " + String.join("; ", errors));
    }

    private static void applyDataExportSolutionSelection(Object feature) {
        if (dataExportTimeValues.length > 0) {
            setIfPossible(feature, "innerinput", "interp");
            setIfPossible(feature, "t", dataExportTimeValues);
            setIfPossible(feature, "time", dataExportTimeValues);
            return;
        }
        if (dataExportSolnums.length > 0) {
            setIfPossible(feature, "innerinput", "manual");
            setIfPossible(feature, "solnum", dataExportSolnums);
            return;
        }
        if (!dataExportInnerInput.trim().isEmpty()) {
            setIfPossible(feature, "innerinput", dataExportInnerInput);
        }
    }

    private static void writeDataExportReport(Path out, String targetDataset, Path target, boolean success, List<String> errors) throws Exception {
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            w.write("{\n");
            w.write("  \"source_kind\": \"external_comsol_particle_export_data_table\",\n");
            w.write("  \"case_name\": " + json(caseName) + ",\n");
            w.write("  \"success\": " + success + ",\n");
            w.write("  \"required\": " + exportDataTableRequired + ",\n");
            w.write("  \"dataset\": " + json(targetDataset) + ",\n");
            w.write("  \"filename\": " + json(target.toString()) + ",\n");
            w.write("  \"innerinput\": " + json(dataExportInnerInput) + ",\n");
            w.write("  \"solnums\": " + jsonIntArray(dataExportSolnums) + ",\n");
            w.write("  \"time_values\": " + jsonNumberArray(dataExportTimeValues) + ",\n");
            w.write("  \"expressions\": " + jsonArray(dataExportExpressions.toArray(new String[0])) + ",\n");
            w.write("  \"errors\": " + jsonArray(errors.toArray(new String[0])) + "\n");
            w.write("}\n");
        }
    }

    private static boolean setIfPossible(Object target, String property, Object value) {
        try {
            call(target, "set", property, value);
            return true;
        } catch (Throwable ignored) {
            return false;
        }
    }

    private static boolean setPhysicsPropertyIfPossible(Object physics, String propertyName, String value) {
        if (setIfPossible(physics, propertyName, value)) {
            return true;
        }
        Object propertyGroup = safeCall(physics, "prop", propertyName);
        if (propertyGroup == null) {
            return false;
        }
        if (setIfPossible(propertyGroup, propertyName, value)) {
            return true;
        }
        if (setIfPossible(propertyGroup, "value", value)) {
            return true;
        }
        return setIfPossible(propertyGroup, "active", value);
    }

    private static String physicsPropertySetting(Object physics, String propertyName) {
        String value = featureSetting(physics, propertyName);
        if (value != null && !value.trim().isEmpty()) {
            return value;
        }
        Object propertyGroup = safeCall(physics, "prop", propertyName);
        if (propertyGroup == null) {
            return "";
        }
        value = featureSetting(propertyGroup, propertyName);
        if (value != null && !value.trim().isEmpty()) {
            return value;
        }
        value = featureSetting(propertyGroup, "value");
        if (value != null && !value.trim().isEmpty()) {
            return value;
        }
        return featureSetting(propertyGroup, "active");
    }

    private static void removeExport(Object exports, String tag) {
        try {
            call(exports, "remove", tag);
        } catch (Throwable ignored) {
        }
    }

    private static double evalFirstFinite(Object interp, double[][] coords) {
        double[] values = evalMany(interp, coords);
        for (double value : values) {
            if (Double.isFinite(value)) {
                return value;
            }
        }
        return Double.NaN;
    }

    private static double[] evalMany(Object interp, double[][] coords) {
        call(interp, "setInterpolationCoordinates", coords);
        Object data = call(interp, "getData");
        return firstVector(data, coords[0].length);
    }

    private static double[][] gridCoordinates() {
        int n = 1;
        for (int d = 0; d < spatialDim; d++) {
            n *= axisCount[d];
        }
        double[][] coords = new double[spatialDim][n];
        int idx = 0;
        if (spatialDim == 1) {
            for (double x : linspace(axisMin[0], axisMax[0], axisCount[0])) {
                coords[0][idx++] = x;
            }
            return coords;
        }
        if (spatialDim == 2) {
            double[] a0 = linspace(axisMin[0], axisMax[0], axisCount[0]);
            double[] a1 = linspace(axisMin[1], axisMax[1], axisCount[1]);
            for (double x : a0) {
                for (double y : a1) {
                    coords[0][idx] = x;
                    coords[1][idx] = y;
                    idx++;
                }
            }
            return coords;
        }
        double[] a0 = linspace(axisMin[0], axisMax[0], axisCount[0]);
        double[] a1 = linspace(axisMin[1], axisMax[1], axisCount[1]);
        double[] a2 = linspace(axisMin[2], axisMax[2], axisCount[2]);
        for (double x : a0) {
            for (double y : a1) {
                for (double z : a2) {
                    coords[0][idx] = x;
                    coords[1][idx] = y;
                    coords[2][idx] = z;
                    idx++;
                }
            }
        }
        return coords;
    }

    private static double[][] probeCoordinates() {
        int n = Math.max(3, spatialDim * 3);
        double[][] coords = new double[spatialDim][n];
        for (int i = 0; i < n; i++) {
            double f = (i + 1.0) / (n + 1.0);
            for (int d = 0; d < spatialDim; d++) {
                coords[d][i] = axisMin[d] + f * (axisMax[d] - axisMin[d]);
            }
        }
        return coords;
    }

    private static double[] linspace(double min, double max, int count) {
        double[] out = new double[count];
        double step = (max - min) / (count - 1);
        for (int i = 0; i < count; i++) {
            out[i] = min + step * i;
        }
        return out;
    }

    private static int sampleContextCount() {
        int count = 1;
        if (solnums.length > count) {
            count = solnums.length;
        }
        if (timeValues.length > count) {
            count = timeValues.length;
        }
        return count;
    }

    private static boolean shouldWriteSampleContextColumns() {
        return solnums.length > 0 || timeValues.length > 0 || sampleContextCount() > 1;
    }

    private static int sampleSolnum(int index) {
        if (solnums.length > 0) {
            return solnums[Math.min(index, solnums.length - 1)];
        }
        return solnum;
    }

    private static double sampleTime(int index) {
        if (timeValues.length > 0) {
            return timeValues[Math.min(index, timeValues.length - 1)];
        }
        return Double.NaN;
    }

    private static int[] activeSolnumsForManifest() {
        if (solnums.length > 0) {
            return solnums;
        }
        if (solnum > 0) {
            return new int[]{solnum};
        }
        return new int[0];
    }

    private static double[] firstVector(Object data, int expected) {
        if (data instanceof double[][][]) {
            double[][][] a = (double[][][]) data;
            return a.length == 0 || a[0].length == 0 ? filledNaN(expected) : padded(a[0][0], expected);
        }
        if (data instanceof double[][]) {
            double[][] a = (double[][]) data;
            return a.length == 0 ? filledNaN(expected) : padded(a[0], expected);
        }
        if (data instanceof double[]) {
            return padded((double[]) data, expected);
        }
        if (data instanceof Double) {
            double[] out = filledNaN(expected);
            out[0] = (Double) data;
            return out;
        }
        return filledNaN(expected);
    }

    private static double[] padded(double[] values, int expected) {
        if (values.length == expected) {
            return values;
        }
        double[] out = filledNaN(expected);
        int n = Math.min(values.length, expected);
        for (int i = 0; i < n; i++) {
            out[i] = values[i];
        }
        return out;
    }

    private static double[] filledNaN(int n) {
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = Double.NaN;
        }
        return out;
    }

    private static double[][] normalizeVertices(double[][] raw) {
        if (raw.length == 0) {
            return raw;
        }
        if (raw.length <= 3 && raw[0].length > raw.length) {
            int sdim = raw.length;
            int n = raw[0].length;
            double[][] out = new double[n][sdim];
            for (int d = 0; d < sdim; d++) {
                for (int i = 0; i < n; i++) {
                    out[i][d] = raw[d][i];
                }
            }
            return out;
        }
        return raw;
    }

    private static int[][] normalizeElements(int[][] raw, int elementCount) {
        if (raw.length == 0 || elementCount == raw.length) {
            return raw;
        }
        if (raw[0].length == elementCount) {
            int nvp = raw.length;
            int[][] out = new int[elementCount][nvp];
            for (int j = 0; j < nvp; j++) {
                for (int i = 0; i < elementCount; i++) {
                    out[i][j] = raw[j][i];
                }
            }
            return out;
        }
        return raw;
    }

    private static Object call(Object target, String name, Object... args) {
        if (target == null) {
            throw new RuntimeException("Cannot call " + name + " on null target");
        }
        Throwable last = null;
        for (Method method : target.getClass().getMethods()) {
            if (!method.getName().equals(name) || method.getParameterCount() != args.length) {
                continue;
            }
            try {
                return method.invoke(target, args);
            } catch (InvocationTargetException t) {
                Throwable cause = t.getCause() == null ? t : t.getCause();
                throw new RuntimeException("Method " + name + " threw: " + cause.getMessage(), cause);
            } catch (Throwable t) {
                last = t;
            }
        }
        throw new RuntimeException("No callable method " + name + " with " + args.length + " args on " + target.getClass(), last);
    }

    private static Object safeCall(Object target, String name, Object... args) {
        try {
            return call(target, name, args);
        } catch (Throwable t) {
            return null;
        }
    }

    private static void writeMethodList(Path path, Object target) {
        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(path, StandardCharsets.UTF_8))) {
            if (target == null) {
                return;
            }
            writer.println(target.getClass().getName());
            for (Method method : target.getClass().getMethods()) {
                Class[] params = method.getParameterTypes();
                List<String> names = new ArrayList<>();
                for (Class param : params) {
                    names.add(param.getName());
                }
                writer.println(method.getName() + "(" + String.join(",", names) + ") -> " + method.getReturnType().getName());
            }
        } catch (Throwable ignored) {
        }
    }

    private static String[] listTags(Object list) {
        return listTagsLike(list, "tags");
    }

    private static String[] listTagsLike(Object list, String method) {
        Object raw = safeCall(list, method);
        if (raw instanceof String[]) {
            return (String[]) raw;
        }
        return new String[0];
    }

    private static int[] selectionEntities(Object selection) {
        if (selection == null) {
            return new int[0];
        }
        for (String method : new String[]{"entities", "inputEntities"}) {
            int[] out = normalizeIntArray(safeCall(selection, method));
            if (out != null) {
                return out;
            }
        }
        return new int[0];
    }

    private static int[] normalizeIntArray(Object value) {
        if (value instanceof int[]) {
            return (int[]) value;
        }
        if (value instanceof Integer[]) {
            Integer[] raw = (Integer[]) value;
            int[] out = new int[raw.length];
            for (int i = 0; i < raw.length; i++) {
                out[i] = raw[i] == null ? 0 : raw[i];
            }
            return out;
        }
        return null;
    }

    private static String stringOrEmpty(Object value) {
        return value == null ? "" : String.valueOf(value);
    }

    private static Map<String, List<String>> parseExpressionLists(String text) {
        Map<String, List<String>> out = new LinkedHashMap<>();
        Matcher matcher = Pattern.compile("\"([A-Za-z0-9_]+)\"\\s*:\\s*\\[(.*?)\\]", Pattern.DOTALL).matcher(text);
        while (matcher.find()) {
            String key = matcher.group(1);
            String body = matcher.group(2);
            List<String> values = new ArrayList<>();
            Matcher strings = Pattern.compile("\"(.*?)\"").matcher(body);
            while (strings.find()) {
                values.add(strings.group(1));
            }
            if (!values.isEmpty()) {
                out.put(key, values);
            }
        }
        return out;
    }

    private static List<String> firstExpressionList(Map<String, List<String>> source, String... keys) {
        for (String key : keys) {
            List<String> values = source.get(key);
            if (values != null && !values.isEmpty()) {
                return new ArrayList<>(values);
            }
        }
        return new ArrayList<>();
    }

    private static double jsonDouble(String text, String key, double fallback) {
        Matcher matcher = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*([-+0-9.eE]+)").matcher(text);
        return matcher.find() ? Double.parseDouble(matcher.group(1)) : fallback;
    }

    private static int[] jsonIntArrayConfig(String text, String key, int[] fallback) {
        Matcher matcher = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*\\[(.*?)\\]", Pattern.DOTALL).matcher(text);
        if (!matcher.find()) {
            return fallback;
        }
        List<Integer> values = new ArrayList<>();
        Matcher numbers = Pattern.compile("[-+0-9]+").matcher(matcher.group(1));
        while (numbers.find()) {
            values.add(Integer.parseInt(numbers.group()));
        }
        if (values.isEmpty()) {
            return fallback;
        }
        int[] out = new int[values.size()];
        for (int i = 0; i < values.size(); i++) {
            out[i] = values.get(i);
        }
        return out;
    }

    private static double[] jsonDoubleArray(String text, String key, double[] fallback) {
        Matcher matcher = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*\\[(.*?)\\]", Pattern.DOTALL).matcher(text);
        if (!matcher.find()) {
            return fallback;
        }
        List<Double> values = new ArrayList<>();
        Matcher numbers = Pattern.compile("[-+0-9.eE]+").matcher(matcher.group(1));
        while (numbers.find()) {
            values.add(Double.parseDouble(numbers.group()));
        }
        if (values.isEmpty()) {
            return fallback;
        }
        double[] out = new double[values.size()];
        for (int i = 0; i < values.size(); i++) {
            out[i] = values.get(i);
        }
        return out;
    }

    private static boolean jsonBoolean(String text, String key, boolean fallback) {
        Matcher matcher = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*(true|false)").matcher(text);
        return matcher.find() ? Boolean.parseBoolean(matcher.group(1)) : fallback;
    }

    private static String jsonString(String text, String key, String fallback) {
        Matcher matcher = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*\"(.*?)\"").matcher(text);
        return matcher.find() ? matcher.group(1) : fallback;
    }

    private static String[] jsonStringArray(String text, String key, String[] fallback) {
        Matcher matcher = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*\\[(.*?)\\]", Pattern.DOTALL).matcher(text);
        if (!matcher.find()) {
            return fallback;
        }
        List<String> values = new ArrayList<>();
        Matcher strings = Pattern.compile("\"(.*?)\"").matcher(matcher.group(1));
        while (strings.find()) {
            values.add(strings.group(1));
        }
        return values.isEmpty() ? fallback : values.toArray(new String[0]);
    }

    private static String[] defaultAxisNames(int dim) {
        if (dim == 1) {
            return new String[]{"x"};
        }
        if (dim == 2) {
            return new String[]{"x", "y"};
        }
        return new String[]{"x", "y", "z"};
    }

    private static String firstNonEmpty(String... values) {
        for (String value : values) {
            if (value != null && !value.trim().isEmpty()) {
                return value;
            }
        }
        return "";
    }

    private static String sanitize(String value) {
        return value.replaceAll("[^A-Za-z0-9_]", "_");
    }

    private static String json(String value) {
        if (value == null) {
            return "null";
        }
        return "\"" + value.replace("\\", "\\\\").replace("\"", "\\\"").replace("\r", "\\r").replace("\n", "\\n") + "\"";
    }

    private static String jsonNumber(double value) {
        return Double.isFinite(value) ? String.format(Locale.US, "%.17g", value) : "null";
    }

    private static String jsonArray(String[] values) {
        List<String> quoted = new ArrayList<>();
        for (String value : values) {
            quoted.add(json(value));
        }
        return "[" + String.join(", ", quoted) + "]";
    }

    private static String jsonIntArray(int[] values) {
        List<String> formatted = new ArrayList<>();
        for (int value : values) {
            formatted.add(String.valueOf(value));
        }
        return "[" + String.join(", ", formatted) + "]";
    }

    private static String jsonNumberArray(double[] values) {
        List<String> formatted = new ArrayList<>();
        for (double value : values) {
            formatted.add(jsonNumber(value));
        }
        return "[" + String.join(", ", formatted) + "]";
    }

    private static String sha256(Path path) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        byte[] bytes = Files.readAllBytes(path);
        byte[] hash = digest.digest(bytes);
        StringBuilder sb = new StringBuilder();
        for (byte b : hash) {
            sb.append(String.format("%02x", b));
        }
        return sb.toString();
    }
}
