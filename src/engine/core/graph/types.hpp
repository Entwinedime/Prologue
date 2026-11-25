#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/contracts/expression.hpp"
#include "core/graph/state.hpp"

namespace prologue {

enum class ResourceKind { Thread, Stream, Service };
enum class WorkKind { Host, Device, Transfer, Communication, Collective, Wait, Boundary };
enum class Relation { Order, Submission, Data, Signal, Control, State, Communication, Observation };
enum class Direction { Input, Output };
enum class PortUse { Control, Data, Metadata, Signal };

// Snapshot accepted commands on the peer stream; host queueing and device submission can finish later.
template <class Boundary> struct StreamWait {
    std::size_t resource;
    Boundary begin;
};

ResourceKind resource_kind(std::string_view name);
std::string_view resource_name(ResourceKind kind);
WorkKind work_kind(std::string_view name);
std::string_view work_name(WorkKind kind);
Relation relation_kind(std::string_view name);
std::string_view relation_name(Relation kind);
Direction direction_kind(std::string_view name);
std::string_view direction_name(Direction direction);
PortUse port_use(std::string_view name);
std::string_view port_use_name(PortUse use);

struct PayloadAgreement {
    std::string port;
    Direction direction;
    bool shape;

    bool operator==(const PayloadAgreement &) const = default;
};

struct PeerRows {
    std::vector<std::int64_t> send;
    std::vector<std::int64_t> receive;

    static PeerRows read(const nlohmann::json & source);
    nlohmann::json document() const;
};

// Participants declare dependencies and any observed tensor fields that must agree across peers.
struct CommunicationGroup {
    std::string channel;
    std::string operation;
    std::vector<std::int64_t> members;
    std::vector<std::int64_t> required;
    std::int64_t rank;
    std::size_t order;
    std::vector<PayloadAgreement> agreements{};
    bool ordered_service = false;
    std::optional<PeerRows> peer_rows{};

    static CommunicationGroup read(const nlohmann::json & source);
    nlohmann::json document() const;
    void bind_rows(PeerRows rows);
    void validate() const;
};

struct ResourceClass {
    ResourceKind kind;
    std::string role;
    bool serial;

    bool operator==(const ResourceClass &) const = default;
    static ResourceClass read(const nlohmann::json & source);
    nlohmann::json document() const;
};

struct Resource : ResourceClass {
    std::int64_t rank;
    std::string identity;
    std::optional<std::string> binding{};

    bool operator==(const Resource &) const = default;

    static Resource read(const nlohmann::json & source);
    nlohmann::json document() const;
};

// Actual geometry is evidence; a rule is a separate declaration for template construction.
struct TensorLayout {
    std::string dtype;
    std::vector<std::int64_t> shape;

    bool operator==(const TensorLayout &) const = default;
};

struct TensorRule {
    std::string dtype;
    std::vector<Expression> shape;
    std::shared_ptr<const RowPacking> packing{};

    static TensorRule read(const nlohmann::json & source);
    nlohmann::json document(const std::map<std::string, Expression> & arguments = {}) const;
    TensorLayout evaluate(const Parameters & parameters) const;
};

// The observed operand belongs to the service. It need not cross the scheduling boundary.
struct TensorCopy {
    std::string operand;
    TensorRule tensor;
    std::int64_t element_bytes;
    std::string caller;
    std::string implementation;
    std::string direction;

    static std::shared_ptr<const TensorCopy> read(const nlohmann::json & work);
    nlohmann::json document() const;
    std::int64_t bytes(const Parameters & parameters) const;
};

struct StorageBinding {
    std::string identity;
    std::int64_t version;
};

struct ScopeInfo {
    std::optional<std::size_t> parent;
    std::string role;
    std::string site;
    std::string implementation;
    Parameters parameters;
    Parameters constants;
    std::map<std::string, Expression> arguments;
    nlohmann::json work;
    nlohmann::json bindings;
    std::vector<Expression> guards{};
    std::shared_ptr<const RowCall> row_call{};

    static ScopeInfo read(const nlohmann::json & source);
    nlohmann::json document() const;
};

struct PackedRows {
    std::shared_ptr<const RowPacking> rule;
    nlohmann::json slices;
};

struct TensorObservation {
    std::string value;
    std::string storage;
    std::int64_t version;
    std::string dtype;
    std::vector<std::int64_t> shape;
    std::optional<std::vector<Expression>> shape_rule;
    std::optional<std::size_t> writer{};

    // Mutation changes the writer, never the allocation's owner.
    std::optional<std::size_t> allocation{};

    // An adapter-declared persistent storage owner, independent of the observing callsite.
    std::string buffer{};

    std::shared_ptr<const PackedRows> packing{};

    static TensorObservation read(const nlohmann::json & source, const ScopeInfo & scope);
    void bind_rows(const std::shared_ptr<const RowPacking> & rule, const ScopeInfo & scope);
    nlohmann::json document() const;
};

} // namespace prologue
