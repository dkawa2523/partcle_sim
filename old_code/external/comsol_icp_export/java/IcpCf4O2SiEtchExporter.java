import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class IcpCf4O2SiEtchExporter {
    private static Path mphPath;
    private static Path configPath;
    private static Path outDir;
    private static String modelName;
    private static String study;
    private static String dataset;
    private static String solution;
    private static int solutionNumber;
    private static String meshTag;
    private static String geometryModelUnit;
    private static double geometryScaleMPerModelUnit;
    private static String parameterName;
    private static String parameterValue;
    private static List<Integer> vacuumDomainIds = new ArrayList<>();
    private static double rMin = 0.0;
    private static double rMax = 0.16;
    private static double zMin = 0.0;
    private static double zMax = 0.20;
    private static int rCount = 241;
    private static int zCount = 301;
    private static Map<String, String> expressions = new LinkedHashMap<>();
    private static Map<String, String> expressionUnits = new LinkedHashMap<>();
    private static int nodeSampleCount = 0;

    public static void main(String[] args) throws Exception {
        loadOptions(args);
        loadConfig(configPath);
        Files.createDirectories(outDir);

        Model model = ModelUtil.load("icp_model", mphPath.toString());
        validateConfiguredProvenance(model);
        tryWriteMethodList(outDir.resolve("result_methods.txt"), safeCall(model, "result"));
        Object datasetList = safeCall(safeCall(model, "result"), "dataset");
        tryWriteMethodList(outDir.resolve("dataset_list_methods.txt"), datasetList);
        tryWriteTags(outDir.resolve("dataset_tags.txt"), datasetList);

        exportMesh(model, outDir.resolve("mesh.mphtxt"));
        tryWriteMaterialInventory(model, outDir.resolve("material_inventory.json"));

        double[] rAxis = linspace(rMin, rMax, rCount);
        double[] zAxis = linspace(zMin, zMax, zCount);
        Map<String, String> selected = validateExpressions(model, probeCoordinates(rAxis, zAxis));

        writeInventory(outDir.resolve("expression_inventory.json"), selected);
        writeFieldSamples(outDir.resolve("field_samples.csv"), model, selected, rAxis, zAxis);
        // The mesh-node table is the artifact the solver samples.  The grid
        // table above stays as a readable reference of the same export.
        nodeSampleCount = writeNodeSamples(outDir.resolve("field_samples_nodes.csv"), model, selected);
        writeManifest(outDir.resolve("export_manifest.json"), selected, rAxis, zAxis);

        try {
            ModelUtil.disconnect();
        } catch (Throwable ignored) {
        }
    }

    private static void loadOptions(String[] args) {
        if (args == null) {
            args = new String[0];
        }
        Map<String, String> map = new LinkedHashMap<>();
        for (int i = 0; i + 1 < args.length; i += 2) {
            map.put(args[i].replaceFirst("^-+", ""), args[i + 1]);
        }
        String mph = firstNonEmpty(map.get("mph"), System.getenv("COMSOL_ICP_MPH"));
        String config = firstNonEmpty(map.get("config"), System.getenv("COMSOL_ICP_CONFIG"));
        String out = firstNonEmpty(map.get("outdir"), System.getenv("COMSOL_ICP_OUTDIR"));
        if (mph == null || config == null || out == null) {
            throw new IllegalArgumentException("mph, config, and outdir must be provided explicitly");
        }
        mphPath = Paths.get(mph);
        configPath = Paths.get(config);
        outDir = Paths.get(out);
    }

    private static void loadConfig(Path path) throws IOException {
        String text = new String(Files.readAllBytes(path), StandardCharsets.UTF_8);
        Map<String, List<String>> configuredExpressions = parseExpressionLists(text);
        configuredExpressions.remove("required");
        for (Map.Entry<String, List<String>> entry : configuredExpressions.entrySet()) {
            if (entry.getValue().size() != 1) {
                throw new IllegalArgumentException(
                    "Each COMSOL semantic quantity must declare exactly one expression; "
                    + entry.getKey() + " has " + entry.getValue().size()
                );
            }
            expressions.put(entry.getKey(), entry.getValue().get(0));
        }
        expressionUnits = parseStringObject(text, "units");
        if (expressions.isEmpty() || !expressions.keySet().equals(expressionUnits.keySet())) {
            throw new IllegalArgumentException("expressions and units must be non-empty and have identical keys");
        }
        modelName = requiredJsonString(text, "model_name");
        study = requiredJsonString(text, "study");
        dataset = requiredJsonString(text, "dataset");
        solution = requiredJsonString(text, "solution");
        solutionNumber = requiredPositiveInt(text, "solution_number");
        meshTag = requiredJsonString(text, "mesh_tag");
        geometryModelUnit = requiredJsonString(text, "geometry_model_unit");
        geometryScaleMPerModelUnit = requiredPositiveDouble(text, "geometry_scale_m_per_model_unit");
        parameterName = requiredJsonString(text, "parameter_name");
        parameterValue = requiredJsonString(text, "parameter_value");
        vacuumDomainIds = requiredPositiveIntList(text, "vacuum_domain_ids");
        rMin = requiredJsonDouble(text, "r_min");
        rMax = requiredJsonDouble(text, "r_max");
        zMin = requiredJsonDouble(text, "z_min");
        zMax = requiredJsonDouble(text, "z_max");
        rCount = requiredPositiveInt(text, "r_count");
        zCount = requiredPositiveInt(text, "z_count");
        if (!(rMax > rMin) || !(zMax > zMin) || rCount < 2 || zCount < 2) {
            throw new IllegalArgumentException("grid bounds must increase and each grid count must be at least two");
        }
    }

    private static void validateConfiguredProvenance(Model model) {
        call(model, "study", study);
        Object configuredSolution = call(model, "sol", solution);
        Object configuredDataset = call(call(model, "result"), "dataset", dataset);
        call(model, "mesh", meshTag);

        requireTagReference(
            "dataset " + dataset + " solution",
            solution,
            (String) call(configuredDataset, "getString", "solution")
        );
        requireTagReference(
            "solution " + solution + " study",
            study,
            (String) call(configuredSolution, "study")
        );
        validateSolutionParameter(model, configuredSolution);
    }

    private static void requireTagReference(String owner, String expected, String actual) {
        if (!expected.equals(actual)) {
            throw new IllegalArgumentException(
                owner + " must be " + expected + ", but the saved model references " + actual
            );
        }
    }

    private static void validateSolutionParameter(Model model, Object configuredSolution) {
        String[] names = (String[]) call(configuredSolution, "getPNames");
        int parameterIndex = -1;
        for (int i = 0; i < names.length; i++) {
            if (parameterName.equals(names[i])) {
                parameterIndex = i;
                break;
            }
        }
        if (parameterIndex < 0) {
            throw new IllegalArgumentException(
                "Saved solution " + solution + " does not contain parameter " + parameterName
            );
        }

        double[] values = (double[]) call(configuredSolution, "getPVals", solutionNumber);
        if (parameterIndex >= values.length) {
            throw new IllegalArgumentException(
                "Saved solution " + solution + " has no parameter value for solution_number="
                + solutionNumber
            );
        }

        Object solutionInfo = call(configuredSolution, "getSolutioninfo");
        String unit = stringOrEmpty(call(solutionInfo, "getUnit", parameterName)).trim();
        String expectedExpression = unit.isEmpty() || "1".equals(unit)
            ? "(" + parameterValue + ")"
            : "((" + parameterValue + ")/(1[" + unit + "]))";
        Object parameters = call(model, "param");
        double expectedValue = ((Number) call(parameters, "evaluate", expectedExpression)).doubleValue();
        double actualValue = values[parameterIndex];
        double tolerance = 1.0e-12 * Math.max(1.0, Math.max(Math.abs(actualValue), Math.abs(expectedValue)));
        if (!Double.isFinite(actualValue)
            || !Double.isFinite(expectedValue)
            || Math.abs(actualValue - expectedValue) > tolerance) {
            throw new IllegalArgumentException(
                "Saved solution parameter mismatch: " + parameterName + "=" + actualValue
                + (unit.isEmpty() ? "" : "[" + unit + "]")
                + " at solution_number=" + solutionNumber
                + ", configured value=" + parameterValue
            );
        }
    }

    private static void exportMesh(Model model, Path out) {
        List<String> errors = new ArrayList<>();
        try {
            Object mesh = call(model, "mesh", meshTag);
            writeMphtxtFromMeshSequence(mesh, out);
            return;
        } catch (Throwable t) {
            errors.add(t.toString());
            tryWriteMethodList(outDir.resolve("mesh_methods.txt"), safeCall(model, "mesh", meshTag));
        }
        for (String compTag : new String[]{"comp1", "comp2"}) {
            try {
                Object comp = call(model, "component", compTag);
                Object mesh = call(comp, "mesh", meshTag);
                call(mesh, "export", out.toString());
                return;
            } catch (Throwable t) {
                errors.add(t.toString());
            }
        }
        tryWriteMethodList(outDir.resolve("model_methods.txt"), model);
        throw new RuntimeException("Could not export mesh.mphtxt through COMSOL Java API. Errors: " + errors);
    }

    private static void writeMphtxtFromMeshSequence(Object mesh, Path out) throws IOException {
        double[][] vertices = normalizeVertices((double[][]) call(mesh, "getVertex"));
        String[] types = (String[]) call(mesh, "getTypes");
        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(out, StandardCharsets.UTF_8))) {
            int sdim = vertices.length == 0 ? 0 : vertices[0].length;
            writer.println(sdim + " # sdim");
            writer.println(vertices.length + " # number of mesh vertices");
            writer.println("# Mesh vertex coordinates");
            for (int i = 0; i < vertices.length; i++) {
                for (int d = 0; d < sdim; d++) {
                    if (d > 0) {
                        writer.print(" ");
                    }
                    writer.print(String.format(Locale.US, "%.17g", vertices[i][d]));
                }
                writer.println();
            }
            writer.println(types.length + " # number of element types");
            int typeIndex = 0;
            for (String type : types) {
                int[] entity = (int[]) call(mesh, "getElemEntity", type);
                int[][] elems = normalizeElements((int[][]) call(mesh, "getElem", type), entity.length);
                int nvp = elems.length == 0 ? 0 : elems[0].length;
                writer.println(typeIndex + " " + type + " # type name");
                writer.println(nvp + " # number of vertices per element");
                writer.println(elems.length + " # number of elements");
                writer.println("# Elements");
                for (int i = 0; i < elems.length; i++) {
                    for (int j = 0; j < nvp; j++) {
                        if (j > 0) {
                            writer.print(" ");
                        }
                        writer.print(elems[i][j]);
                    }
                    writer.println();
                }
                writer.println(entity.length + " # number of geometric entity indices");
                writer.println("# Geometric entity indices");
                for (int i = 0; i < entity.length; i++) {
                    writer.println(entity[i]);
                }
                typeIndex++;
            }
        }
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

    private static Map<String, String> validateExpressions(Model model, double[][] probeCoords) {
        Map<String, String> selected = new LinkedHashMap<>();
        for (Map.Entry<String, String> entry : expressions.entrySet()) {
            String key = entry.getKey();
            String expr = entry.getValue();
            String tag = "inv_" + sanitize(key);
            try {
                Object interp = createInterp(model, tag, expr, expressionUnits.get(key));
                double value = evalFirstFinite(interp, probeCoords);
                if (!Double.isFinite(value)) {
                    throw new RuntimeException("configured expression has no finite probe value");
                }
                selected.put(key, expr);
            } catch (Throwable t) {
                throw new RuntimeException(
                    "Configured COMSOL expression failed for " + key + " on dataset=" + dataset
                    + ", solution=" + solution + ", solution_number=" + solutionNumber
                    + ", expression=" + expr + ", unit=" + expressionUnits.get(key),
                    t
                );
            } finally {
                removeNumerical(model, tag);
            }
        }
        return selected;
    }

    private static Object createInterp(Model model, String tag, String expr, String unit) {
        Object result = call(model, "result");
        Object numerical = call(result, "numerical");
        try {
            call(numerical, "remove", tag);
        } catch (Throwable ignored) {
        }
        call(numerical, "create", tag, "Interp");
        Object interp = call(result, "numerical", tag);
        call(interp, "set", "data", dataset);
        call(interp, "set", "solnum", new int[]{solutionNumber});
        call(interp, "set", "expr", new String[]{expr});
        call(interp, "set", "unit", new String[]{unit});
        restrictToVacuumDomains(interp);
        tryWriteMethodList(outDir.resolve("numerical_feature_methods.txt"), interp);
        return interp;
    }

    /**
     * Evaluate only inside the explicit vacuum-domain selection.
     *
     * Without a selection an Interp feature evaluates over every domain of the
     * dataset, so a point on a vacuum/solid interface can be answered from the
     * solid side and return NaN for a plasma-only expression.  Restricting the
     * selection makes every mesh vertex of the particle domain return its
     * vacuum-side value, which is what lets the case builder require a finite
     * value at every node instead of carrying a near-wall fallback.
     */
    private static void restrictToVacuumDomains(Object interp) {
        if (vacuumDomainIds.isEmpty()) {
            throw new RuntimeException(
                "vacuum_domain_ids must be non-empty: an export cannot resolve "
                + "vacuum/solid interface nodes without an explicit selection"
            );
        }
        int[] ids = new int[vacuumDomainIds.size()];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = vacuumDomainIds.get(i).intValue();
        }
        Object selection = call(interp, "selection");
        call(selection, "geom", 2);
        call(selection, "set", ids);
    }

    /**
     * Read mesh vertex coordinates in the order mesh.mphtxt writes them.
     *
     * The mphtxt vertex block and the API vertex array come from the same mesh
     * object, so the row index is an exact identity for joining node values to
     * mesh topology.  No coordinate rounding is involved.
     */
    private static double[][] meshVertexCoordinates(Model model) {
        Object mesh = call(call(model, "mesh"), meshTag);
        Object raw = call(mesh, "getVertex");
        if (!(raw instanceof double[][])) {
            throw new RuntimeException(
                "COMSOL mesh getVertex did not return a coordinate matrix for mesh tag " + meshTag
            );
        }
        double[][] vertices = (double[][]) raw;
        if (vertices.length < 2 || vertices[0].length == 0) {
            throw new RuntimeException("COMSOL mesh " + meshTag + " reported no vertices");
        }
        return vertices;
    }

    private static int writeNodeSamples(Path out, Model model, Map<String, String> selected) throws IOException {
        double[][] vertices = meshVertexCoordinates(model);
        int nodeCount = vertices[0].length;
        double[][] coords = new double[2][nodeCount];
        System.arraycopy(vertices[0], 0, coords[0], 0, nodeCount);
        System.arraycopy(vertices[1], 0, coords[1], 0, nodeCount);

        Map<String, Object> features = new LinkedHashMap<>();
        Map<String, double[]> valuesByKey = new LinkedHashMap<>();
        try {
            for (Map.Entry<String, String> entry : selected.entrySet()) {
                features.put(
                    entry.getKey(),
                    createInterp(
                        model,
                        "node_" + sanitize(entry.getKey()),
                        entry.getValue(),
                        expressionUnits.get(entry.getKey())
                    )
                );
            }
            for (String key : selected.keySet()) {
                valuesByKey.put(key, evalMany(features.get(key), coords));
            }
        } finally {
            for (String key : selected.keySet()) {
                removeNumerical(model, "node_" + sanitize(key));
            }
        }

        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(out, StandardCharsets.UTF_8))) {
            writer.print("node_index,r,z");
            for (String key : selected.keySet()) {
                writer.print(",");
                writer.print(key);
            }
            writer.println();
            for (int i = 0; i < nodeCount; i++) {
                writer.printf(Locale.US, "%d,%.17g,%.17g", i, coords[0][i], coords[1][i]);
                for (String key : selected.keySet()) {
                    double value = valuesByKey.get(key)[i];
                    writer.print(",");
                    writer.print(Double.isFinite(value) ? String.format(Locale.US, "%.17g", value) : "NaN");
                }
                writer.println();
            }
        }
        return nodeCount;
    }

    private static void removeNumerical(Model model, String tag) {
        try {
            call(call(call(model, "result"), "numerical"), "remove", tag);
        } catch (Throwable ignored) {
        }
    }

    private static double evalFirstFinite(Object interp, double[][] coords) {
        double[] values = evalMany(interp, coords);
        for (int i = 0; i < values.length; i++) {
            if (Double.isFinite(values[i])) {
                return values[i];
            }
        }
        return Double.NaN;
    }

    private static double[] evalMany(Object interp, double[][] coords) {
        call(interp, "setInterpolationCoordinates", coords);
        Object data = call(interp, "getData");
        return firstVector(data, coords[0].length);
    }

    private static void writeFieldSamples(Path out, Model model, Map<String, String> selected, double[] rAxis, double[] zAxis) throws IOException {
        Map<String, Object> features = new LinkedHashMap<>();
        for (Map.Entry<String, String> entry : selected.entrySet()) {
            features.put(
                entry.getKey(),
                createInterp(
                    model,
                    "grid_" + sanitize(entry.getKey()),
                    entry.getValue(),
                    expressionUnits.get(entry.getKey())
                )
            );
        }
        double[][] coords = gridCoordinates(rAxis, zAxis);
        Map<String, double[]> valuesByKey = new LinkedHashMap<>();
        for (String key : selected.keySet()) {
            valuesByKey.put(key, evalMany(features.get(key), coords));
        }
        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(out, StandardCharsets.UTF_8))) {
            writer.print("r,z,valid_mask");
            for (String key : selected.keySet()) {
                writer.print(",");
                writer.print(key);
            }
            writer.println();
            int idx = 0;
            for (double r : rAxis) {
                for (double z : zAxis) {
                    boolean valid = true;
                    for (String key : selected.keySet()) {
                        double value = valuesByKey.get(key)[idx];
                        if (!Double.isFinite(value)) {
                            valid = false;
                        }
                    }
                    writer.printf(Locale.US, "%.17g,%.17g,%d", r, z, valid ? 1 : 0);
                    for (String key : selected.keySet()) {
                        double value = valuesByKey.get(key)[idx];
                        writer.print(",");
                        writer.print(Double.isFinite(value) ? String.format(Locale.US, "%.17g", value) : "NaN");
                    }
                    writer.println();
                    idx++;
                }
            }
        } finally {
            for (String key : selected.keySet()) {
                removeNumerical(model, "grid_" + sanitize(key));
            }
        }
    }

    private static void writeInventory(Path out, Map<String, String> selected) throws IOException {
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            w.write("{\n");
            w.write("  \"study\": " + json(study) + ",\n");
            w.write("  \"dataset\": " + json(dataset) + ",\n");
            w.write("  \"solution\": " + json(solution) + ",\n");
            w.write("  \"solution_number\": " + solutionNumber + ",\n");
            w.write("  \"selected\": {\n");
            int i = 0;
            for (Map.Entry<String, String> entry : expressions.entrySet()) {
                if (i++ > 0) {
                    w.write(",\n");
                }
                String key = entry.getKey();
                w.write("    " + json(key) + ": {");
                w.write("\"expression\": " + json(selected.get(key)) + ", ");
                w.write("\"unit\": " + json(expressionUnits.get(key)) + ", ");
                w.write("\"dataset\": " + json(dataset) + ", ");
                w.write("\"available\": true");
                w.write("}");
            }
            w.write("\n  }\n");
            w.write("}\n");
        }
    }

    private static void writeManifest(Path out, Map<String, String> selected, double[] rAxis, double[] zAxis) throws Exception {
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            w.write("{\n");
            w.write("  \"source_kind\": \"comsol_java_api_external_export\",\n");
            w.write("  \"mph_path\": " + json(mphPath.toString()) + ",\n");
            w.write("  \"mph_sha256\": " + json(sha256(mphPath)) + ",\n");
            w.write("  \"config_sha256\": " + json(sha256(configPath)) + ",\n");
            w.write("  \"mesh_sha256\": " + json(sha256(outDir.resolve("mesh.mphtxt"))) + ",\n");
            w.write("  \"field_samples_sha256\": " + json(sha256(outDir.resolve("field_samples.csv"))) + ",\n");
            w.write("  \"field_node_samples_sha256\": " + json(sha256(outDir.resolve("field_samples_nodes.csv"))) + ",\n");
            w.write("  \"field_node_sample_count\": " + nodeSampleCount + ",\n");
            w.write("  \"field_node_identity\": \"comsol_mesh_vertex_index\",\n");
            w.write("  \"comsol_version\": " + json(readComsolVersion()) + ",\n");
            w.write("  \"model_name\": " + json(modelName) + ",\n");
            w.write("  \"study\": " + json(study) + ",\n");
            w.write("  \"dataset\": " + json(dataset) + ",\n");
            w.write("  \"solution\": " + json(solution) + ",\n");
            w.write("  \"solution_number\": " + solutionNumber + ",\n");
            w.write("  \"mesh_tag\": " + json(meshTag) + ",\n");
            w.write("  \"parameter_name\": " + json(parameterName) + ",\n");
            w.write("  \"parameter_value\": " + json(parameterValue) + ",\n");
            w.write("  \"vacuum_domain_ids\": " + jsonIntArray(vacuumDomainIds) + ",\n");
            w.write("  \"geometry_model_unit\": " + json(geometryModelUnit) + ",\n");
            w.write("  \"geometry_scale_m_per_model_unit\": " + jsonNumber(geometryScaleMPerModelUnit) + ",\n");
            w.write("  \"solver_coordinate_unit\": \"m\",\n");
            w.write("  \"grid_shape\": [" + rAxis.length + ", " + zAxis.length + "],\n");
            w.write("  \"r_bounds\": [" + jsonNumber(rAxis[0]) + ", " + jsonNumber(rAxis[rAxis.length - 1]) + "],\n");
            w.write("  \"z_bounds\": [" + jsonNumber(zAxis[0]) + ", " + jsonNumber(zAxis[zAxis.length - 1]) + "],\n");
            w.write("  \"r_bounds_model_units\": [" + jsonNumber(rAxis[0]) + ", " + jsonNumber(rAxis[rAxis.length - 1]) + "],\n");
            w.write("  \"z_bounds_model_units\": [" + jsonNumber(zAxis[0]) + ", " + jsonNumber(zAxis[zAxis.length - 1]) + "],\n");
            w.write("  \"r_bounds_m\": [" + jsonNumber(rAxis[0] * geometryScaleMPerModelUnit) + ", " + jsonNumber(rAxis[rAxis.length - 1] * geometryScaleMPerModelUnit) + "],\n");
            w.write("  \"z_bounds_m\": [" + jsonNumber(zAxis[0] * geometryScaleMPerModelUnit) + ", " + jsonNumber(zAxis[zAxis.length - 1] * geometryScaleMPerModelUnit) + "],\n");
            w.write("  \"expression_mapping\": {\n");
            int i = 0;
            for (Map.Entry<String, String> entry : selected.entrySet()) {
                if (i++ > 0) {
                    w.write(",\n");
                }
                w.write("    " + json(entry.getKey()) + ": " + json(entry.getValue()));
            }
            w.write("\n  },\n");
            w.write("  \"expression_units\": {\n");
            int j = 0;
            for (Map.Entry<String, String> entry : expressionUnits.entrySet()) {
                if (j++ > 0) {
                    w.write(",\n");
                }
                w.write("    " + json(entry.getKey()) + ": " + json(entry.getValue()));
            }
            w.write("\n  }\n");
            w.write("}\n");
        }
    }

    private static void tryWriteMaterialInventory(Model model, Path out) {
        try {
            writeMaterialInventory(model, out);
        } catch (Throwable t) {
            try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
                w.write("{\n");
                w.write("  \"source_kind\": \"comsol_material_inventory\",\n");
                w.write("  \"status\": \"unavailable\",\n");
                w.write("  \"error\": " + json(t.getClass().getSimpleName() + ": " + t.getMessage()) + ",\n");
                w.write("  \"materials\": []\n");
                w.write("}\n");
            } catch (Throwable ignored) {
            }
        }
    }

    private static void writeMaterialInventory(Model model, Path out) throws IOException {
        Object materialList = call(model, "material");
        tryWriteMethodList(outDir.resolve("material_list_methods.txt"), materialList);
        String[] tags = (String[]) call(materialList, "tags");
        try (BufferedWriter w = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            w.write("{\n");
            w.write("  \"source_kind\": \"comsol_material_inventory\",\n");
            w.write("  \"status\": \"ok\",\n");
            w.write("  \"entity_id_base\": \"comsol_selection_entities_as_reported\",\n");
            w.write("  \"materials\": [\n");
            for (int i = 0; i < tags.length; i++) {
                String tag = tags[i];
                Object material = call(model, "material", tag);
                if (i == 0) {
                    tryWriteMethodList(outDir.resolve("material_methods.txt"), material);
                }
                Object selection = safeCall(material, "selection");
                if (i == 0 && selection != null) {
                    tryWriteMethodList(outDir.resolve("material_selection_methods.txt"), selection);
                }
                if (i > 0) {
                    w.write(",\n");
                }
                w.write("    {\n");
                w.write("      \"tag\": " + json(tag) + ",\n");
                w.write("      \"label\": " + json(stringOrEmpty(safeCall(material, "label"))) + ",\n");
                w.write("      \"name\": " + json(stringOrEmpty(safeCall(material, "name"))) + ",\n");
                w.write("      \"selection_entities\": " + jsonIntArray(selectionEntities(selection)) + "\n");
                w.write("    }");
            }
            w.write("\n  ]\n");
            w.write("}\n");
        }
    }

    private static int[] selectionEntities(Object selection) {
        if (selection == null) {
            return new int[0];
        }
        for (String method : new String[]{"entities", "inputEntities"}) {
            try {
                int[] out = normalizeIntArray(call(selection, method));
                if (out != null) {
                    return out;
                }
            } catch (Throwable ignored) {
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

    private static Object call(Object target, String name, Object... args) {
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

    private static void tryWriteMethodList(Path path, Object target) {
        if (target == null) {
            return;
        }
        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(path, StandardCharsets.UTF_8))) {
            writer.println(target.getClass().getName());
            for (Method method : target.getClass().getMethods()) {
                Class[] params = method.getParameterTypes();
                List<String> names = new ArrayList<>();
                for (int i = 0; i < params.length; i++) {
                    names.add(params[i].getName());
                }
                writer.println(method.getName() + "(" + String.join(",", names) + ") -> " + method.getReturnType().getName());
            }
        } catch (Throwable ignored) {
        }
    }

    private static void tryWriteTags(Path path, Object list) {
        if (list == null) {
            return;
        }
        try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(path, StandardCharsets.UTF_8))) {
            String[] tags = (String[]) call(list, "tags");
            for (int i = 0; i < tags.length; i++) {
                writer.println(tags[i]);
            }
        } catch (Throwable ignored) {
        }
    }

    private static double firstDouble(Object data) {
        if (data instanceof Double) {
            return (Double) data;
        }
        if (data instanceof double[]) {
            double[] a = (double[]) data;
            return a.length == 0 ? Double.NaN : a[0];
        }
        if (data instanceof double[][]) {
            double[][] a = (double[][]) data;
            return a.length == 0 || a[0].length == 0 ? Double.NaN : a[0][0];
        }
        if (data instanceof double[][][]) {
            double[][][] a = (double[][][]) data;
            return a.length == 0 || a[0].length == 0 || a[0][0].length == 0 ? Double.NaN : a[0][0][0];
        }
        throw new RuntimeException("Cannot extract double from " + data.getClass());
    }

    private static double[] firstVector(Object data, int expected) {
        if (data instanceof double[][][]) {
            double[][][] a = (double[][][]) data;
            if (a.length == 0 || a[0].length == 0) {
                return filledNaN(expected);
            }
            return padded(a[0][0], expected);
        }
        if (data instanceof double[][]) {
            double[][] a = (double[][]) data;
            if (a.length == 0) {
                return filledNaN(expected);
            }
            return padded(a[0], expected);
        }
        if (data instanceof double[]) {
            return padded((double[]) data, expected);
        }
        return new double[]{firstDouble(data)};
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

    private static double[][] gridCoordinates(double[] rAxis, double[] zAxis) {
        int n = rAxis.length * zAxis.length;
        double[][] coords = new double[2][n];
        int idx = 0;
        for (int i = 0; i < rAxis.length; i++) {
            for (int j = 0; j < zAxis.length; j++) {
                coords[0][idx] = rAxis[i];
                coords[1][idx] = zAxis[j];
                idx++;
            }
        }
        return coords;
    }

    private static double[][] probeCoordinates(double[] rAxis, double[] zAxis) {
        int[] frac = new int[]{1, 2, 3};
        double[][] coords = new double[2][frac.length * frac.length];
        int idx = 0;
        for (int a = 0; a < frac.length; a++) {
            for (int b = 0; b < frac.length; b++) {
                coords[0][idx] = rAxis[(rAxis.length - 1) * frac[a] / 4];
                coords[1][idx] = zAxis[(zAxis.length - 1) * frac[b] / 4];
                idx++;
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

    private static Map<String, String> parseStringObject(String text, String key) {
        Matcher object = Pattern.compile(
            "\\\"" + Pattern.quote(key) + "\\\"\\s*:\\s*\\{(.*?)\\}",
            Pattern.DOTALL
        ).matcher(text);
        if (!object.find()) {
            throw new IllegalArgumentException("Missing JSON object: " + key);
        }
        Map<String, String> out = new LinkedHashMap<>();
        Matcher entries = Pattern.compile("\\\"([^\\\"]+)\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"").matcher(object.group(1));
        while (entries.find()) {
            out.put(entries.group(1), entries.group(2));
        }
        return out;
    }

    private static double requiredJsonDouble(String text, String key) {
        Matcher matcher = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*([-+0-9.eE]+)").matcher(text);
        if (!matcher.find()) {
            throw new IllegalArgumentException("Missing numeric JSON value: " + key);
        }
        double value = Double.parseDouble(matcher.group(1));
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException("JSON value must be finite: " + key);
        }
        return value;
    }

    private static double requiredPositiveDouble(String text, String key) {
        double value = requiredJsonDouble(text, key);
        if (!(value > 0.0)) {
            throw new IllegalArgumentException("JSON value must be positive: " + key);
        }
        return value;
    }

    private static int requiredPositiveInt(String text, String key) {
        double value = requiredPositiveDouble(text, key);
        if (value != Math.rint(value) || value > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("JSON value must be a positive integer: " + key);
        }
        return (int) value;
    }

    private static String requiredJsonString(String text, String key) {
        Matcher matcher = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*\"(.*?)\"").matcher(text);
        if (!matcher.find() || matcher.group(1).trim().isEmpty()) {
            throw new IllegalArgumentException("Missing non-empty JSON string: " + key);
        }
        return matcher.group(1);
    }

    private static List<Integer> requiredPositiveIntList(String text, String key) {
        Matcher matcher = Pattern.compile(
            "\\\"" + Pattern.quote(key) + "\\\"\\s*:\\s*\\[(.*?)\\]",
            Pattern.DOTALL
        ).matcher(text);
        if (!matcher.find()) {
            throw new IllegalArgumentException("Missing integer JSON list: " + key);
        }
        List<Integer> out = new ArrayList<>();
        String body = matcher.group(1).trim();
        if (!body.isEmpty()) {
            for (String token : body.split(",")) {
                int value = Integer.parseInt(token.trim());
                if (value <= 0 || out.contains(value)) {
                    throw new IllegalArgumentException(key + " must contain unique positive integers");
                }
                out.add(value);
            }
        }
        if (out.isEmpty()) {
            throw new IllegalArgumentException(
                key + " must explicitly identify the COMSOL domains occupied by particles"
            );
        }
        return out;
    }

    private static String readComsolVersion() {
        try {
            Method method = ModelUtil.class.getMethod("getComsolVersion");
            Object value = method.invoke(null);
            String text = value == null ? "" : value.toString().trim();
            if (text.isEmpty()) {
                throw new IllegalStateException("COMSOL version is empty");
            }
            return text;
        } catch (Throwable t) {
            throw new RuntimeException("Could not read COMSOL version for export provenance", t);
        }
    }

    private static String firstNonEmpty(String... values) {
        for (String value : values) {
            if (value != null && !value.isEmpty()) {
                return value;
            }
        }
        return null;
    }

    private static String sanitize(String value) {
        return value.replaceAll("[^A-Za-z0-9_]", "_");
    }

    private static String json(String value) {
        if (value == null) {
            return "null";
        }
        return "\"" + value.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
    }

    private static String jsonNumber(double value) {
        return Double.isFinite(value) ? String.format(Locale.US, "%.17g", value) : "null";
    }

    private static String jsonIntArray(List<Integer> values) {
        List<String> encoded = new ArrayList<>();
        for (Integer value : values) {
            encoded.add(Integer.toString(value));
        }
        return "[" + String.join(", ", encoded) + "]";
    }

    private static String jsonIntArray(int[] values) {
        List<String> formatted = new ArrayList<>();
        for (int value : values) {
            formatted.add(String.valueOf(value));
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
