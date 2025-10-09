"""
MongoDB Database - Infrastructure Layer

This module provides a MongoDB database client for interacting with MongoDB.
It handles connection, collections, and basic CRUD operations.
"""

from typing import Any, Dict, List, Optional, TypeVar

import pymongo.errors
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

T = TypeVar("T")


class MongoDatabase:
    """MongoDB database client."""

    def __init__(self, mongo_uri: str, db_name: str):
        """
        Initialize the MongoDB database client.

        Args:
            mongo_uri: MongoDB connection URI
            db_name: Name of the database to use
        """
        self.client: MongoClient = MongoClient(mongo_uri)
        self.db: Database = self.client[db_name]

    def get_collection(self, collection_name: str) -> Collection:
        """
        Get a collection from the database.

        Args:
            collection_name: Name of the collection

        Returns:
            MongoDB collection
        """
        return self.db[collection_name]

    async def find_one(
        self, collection_name: str, query: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Find a single document in a collection.

        Args:
            collection_name: Name of the collection
            query: Query to match documents

        Returns:
            The document if found, None otherwise
        """
        return self.db[collection_name].find_one(query)

    async def find_many(
        self,
        collection_name: str,
        query: Dict[str, Any],
        sort_by: Optional[str] = None,
        sort_direction: int = 1,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Find multiple documents in a collection.

        Args:
            collection_name: Name of the collection
            query: Query to match documents
            sort_by: Field to sort by
            sort_direction: Sort direction (1 for ascending, -1 for descending)
            skip: Number of documents to skip
            limit: Maximum number of documents to return

        Returns:
            List of documents
        """
        cursor = self.db[collection_name].find(query)

        if sort_by:
            cursor = cursor.sort(sort_by, sort_direction)

        cursor = cursor.skip(skip).limit(limit)

        return list(cursor)

    async def insert_one(
        self, collection_name: str, document: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Insert a document into a collection.

        Args:
            collection_name: Name of the collection
            document: Document to insert

        Returns:
            The inserted document with any generated fields

        Raises:
            Exception: If the insert fails
        """
        result = self.db[collection_name].insert_one(document)
        if not result.acknowledged:
            raise Exception(f"Failed to insert document in {collection_name}")
        return document

    async def replace_one(
        self, collection_name: str, query: Dict[str, Any], document: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Replace a document in a collection.

        Args:
            collection_name: Name of the collection
            query: Query to match document to replace
            document: New document

        Returns:
            The new document

        Raises:
            Exception: If the document does not exist or the replace fails
        """
        result = self.db[collection_name].replace_one(query, document)
        if result.matched_count == 0:
            raise Exception(f"Document not found in {collection_name}")
        if not result.acknowledged:
            raise Exception(f"Failed to replace document in {collection_name}")
        return document

    async def delete_one(self, collection_name: str, query: Dict[str, Any]) -> None:
        """
        Delete a document from a collection.

        Args:
            collection_name: Name of the collection
            query: Query to match document to delete

        Raises:
            Exception: If the document does not exist or the delete fails
        """
        result = self.db[collection_name].delete_one(query)
        if result.deleted_count == 0:
            raise Exception(f"Document not found in {collection_name}")
        if not result.acknowledged:
            raise Exception(f"Failed to delete document in {collection_name}")

    def close(self) -> None:
        """Close the database connection."""
        self.client.close()

    def _safe_drop_index(self, collection_name: str, index_name: str) -> None:
        """
        Safely drop an index if it exists.

        Args:
            collection_name: Name of the collection
            index_name: Name of the index to drop
        """
        try:
            self.db[collection_name].drop_index(index_name)
        except pymongo.errors.OperationFailure:
            # Index doesn't exist, nothing to do
            pass

    async def create_indexes(self) -> None:
        """
        Create all necessary indexes for the application.
        This is an async method to be called during application startup.
        """
        collection_name = "models"

        # First drop any potentially conflicting indexes
        self._safe_drop_index(collection_name, "created_at_idx")
        self._safe_drop_index(collection_name, "status_idx")
        self._safe_drop_index(collection_name, "model_type_idx")
        self._safe_drop_index(collection_name, "entity_id_idx")
        self._safe_drop_index(collection_name, "feature_idx")
        self._safe_drop_index(collection_name, "model_type_status_idx")
        self._safe_drop_index(collection_name, "entity_feature_idx")

        try:
            # Create indexes for models collection
            self.db[collection_name].create_index("created_at", name="created_at_idx")
            self.db[collection_name].create_index("status", name="status_idx")
            self.db[collection_name].create_index("model_type", name="model_type_idx")
            self.db[collection_name].create_index("entity_id", name="entity_id_idx")
            self.db[collection_name].create_index("feature", name="feature_idx")

            # Create compound indexes for common query combinations
            self.db[collection_name].create_index(
                [("model_type", 1), ("status", 1)],
                name="model_type_status_idx",
                background=True,
            )

            self.db[collection_name].create_index(
                [("entity_id", 1), ("feature", 1)],
                name="entity_feature_idx",
                background=True,
            )
        except pymongo.errors.OperationFailure as e:
            print(f"Warning: Failed to create indexes: {str(e)}")

        training_collection = "training_jobs"
        self._safe_drop_index(training_collection, "prediction_enabled_idx")
        self._safe_drop_index(training_collection, "next_prediction_at_idx")

        try:
            self.db[training_collection].create_index(
                "prediction_config.enabled",
                name="prediction_enabled_idx",
                background=True,
            )
            self.db[training_collection].create_index(
                "next_prediction_at",
                name="next_prediction_at_idx",
                background=True,
            )
        except pymongo.errors.OperationFailure as e:
            print(f"Warning: Failed to create training indexes: {str(e)}")
